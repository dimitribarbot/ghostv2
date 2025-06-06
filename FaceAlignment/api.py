import warnings
from enum import IntEnum
from packaging import version
import importlib

import numpy as np
import torch
from safetensors.torch import load_file

from .models import FAN
from .utils import *

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(root_dir, "weights", "FaceAlignment")


class LandmarksType(IntEnum):
    """Enum class defining the type of landmarks to detect.

    ``TWO_D`` - the detected points ``(x,y)`` are detected in a 2D space and follow the visible contour of the face
    ``TWO_HALF_D`` - this points represent the projection of the 3D points into 3D
    ``THREE_D`` - detect the points ``(x,y,z)``` in a 3D space

    """
    TWO_D = 1
    TWO_HALF_D = 2
    THREE_D = 3


class NetworkSize(IntEnum):
    # TINY = 1
    # SMALL = 2
    # MEDIUM = 3
    LARGE = 4


default_model_urls = {
    '2DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4-cd938726ad.zip',
    '3DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/3DFAN4-4a694010b9.zip',
    'depth': 'https://www.adrianbulat.com/downloads/python-fan/depth-6c4283c0e0.zip',
}

models_urls = {
    '1.6': {
        '2DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4_1.6-c827573f02.zip',
        '3DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/3DFAN4_1.6-ec5cf40a1d.zip',
        'depth': 'https://www.adrianbulat.com/downloads/python-fan/depth_1.6-2aa3f18772.zip',
    },
    '1.5': {
        '2DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4_1.5-a60332318a.zip',
        '3DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/3DFAN4_1.5-176570af4d.zip',
        'depth': 'https://www.adrianbulat.com/downloads/python-fan/depth_1.5-bc10f98e39.zip',
    },
}


class FaceAlignment:
    def __init__(self, landmarks_type, network_size=NetworkSize.LARGE, safetensors_file_path=None,
                 device='cuda', dtype=torch.float32, flip_input=False, face_detector='retinaface', face_detector_kwargs=None, verbose=False):
        self.device = device
        self.flip_input = flip_input
        self.landmarks_type = landmarks_type
        self.verbose = verbose
        self.dtype = dtype
        self.use_safetensors = safetensors_file_path is not None

        if version.parse(torch.__version__) < version.parse('1.5.0'):
            raise ImportError(f'Unsupported pytorch version detected. Minimum supported version of pytorch: 1.5.0\
                            Either upgrade (recommended) your pytorch setup, or downgrade to face-alignment 1.2.0')

        network_size = int(network_size)
        pytorch_version = torch.__version__
        if 'dev' in pytorch_version:
            pytorch_version = pytorch_version.rsplit('.', 2)[0]
        else:
            pytorch_version = pytorch_version.rsplit('.', 1)[0]

        #if 'cuda' in device:
        #    torch.backends.cudnn.benchmark = True

        # Get the face detector
        package_directory_name = os.path.basename(root_dir)
        face_detector_module = importlib.import_module('FaceAlignment.detection.' + face_detector, package=package_directory_name)


        face_detector_kwargs = face_detector_kwargs or {}
        self.face_detector = face_detector_module.FaceDetector(device=device, verbose=verbose, **face_detector_kwargs)

        # Initialise the face alignemnt networks
        if landmarks_type == LandmarksType.TWO_D:
            network_name = '2DFAN-' + str(network_size)
        else:
            network_name = '3DFAN-' + str(network_size)
        if self.use_safetensors:
            self.face_alignment_net = FAN(network_size)
            self.face_alignment_net.load_state_dict(load_file(safetensors_file_path), strict=True)
        else:
            self.face_alignment_net = torch.jit.load(
                load_file_from_url(models_urls.get(pytorch_version, default_model_urls)[network_name], model_dir))

        self.face_alignment_net.to(device, dtype=dtype)
        self.face_alignment_net.eval()

        # Initialiase the depth prediciton network
        if landmarks_type == LandmarksType.THREE_D:
            self.depth_prediciton_net = torch.jit.load(
                load_file_from_url(models_urls.get(pytorch_version, default_model_urls)['depth'], model_dir))

            self.depth_prediciton_net.to(device, dtype=dtype)
            self.depth_prediciton_net.eval()

    @torch.no_grad()
    def get_landmarks_from_image(self, image_or_path, detected_faces=None, return_bboxes=False,
                                 return_landmark_score=False):
        """Predict the landmarks for each face present in the image.

        This function predicts a set of 68 2D or 3D images, one for each image present.
        If detect_faces is None the method will also run a face detector.

         Arguments:
            image_or_path {string or numpy.array or torch.tensor} -- The input image or path to it.

        Keyword Arguments:
            detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
            in the image (default: {None})
            return_bboxes {boolean} -- If True, return the face bounding boxes in addition to the keypoints.
            return_landmark_score {boolean} -- If True, return the keypoint scores along with the keypoints.

        Return:
            result:
                1. if both return_bboxes and return_landmark_score are False, result will be:
                    landmark
                2. Otherwise, result will be one of the following, depending on the actual value of return_* arguments.
                    (landmark, landmark_score, detected_face)
                    (landmark, None,           detected_face)
                    (landmark, landmark_score, None         )
        """
        image = get_image(image_or_path)

        if detected_faces is None:
            detected_faces = self.face_detector.detect_from_image(image.copy())

        if len(detected_faces) == 0:
            warnings.warn("No faces were detected.")
            if return_bboxes or return_landmark_score:
                return None, None, None
            else:
                return None

        landmarks = []
        landmarks_scores = []
        for i, d in enumerate(detected_faces):
            center = torch.tensor(
                [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
            center[1] = center[1] - (d[3] - d[1]) * 0.12
            scale = (d[2] - d[0] + d[3] - d[1]) / self.face_detector.reference_scale

            inp = crop(image, center, scale)
            inp = torch.from_numpy(inp.transpose(
                (2, 0, 1))).float()

            inp = inp.to(self.device, dtype=self.dtype)
            inp.div_(255.0).unsqueeze_(0)

            if self.use_safetensors:
                out = self.face_alignment_net(inp)[-1].detach()
            else:
                out = self.face_alignment_net(inp).detach()
            if self.flip_input:
                if self.use_safetensors:
                    out += flip(self.face_alignment_net(flip(inp))[-1].detach(), is_label=True)
                else:
                    out += flip(self.face_alignment_net(flip(inp)).detach(), is_label=True)
            out = out.to(device='cpu', dtype=torch.float32).numpy()

            pts, pts_img, scores = get_preds_fromhm(out, center.numpy(), scale)
            pts, pts_img = torch.from_numpy(pts), torch.from_numpy(pts_img)
            pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)
            scores = scores.squeeze(0)

            if self.landmarks_type == LandmarksType.THREE_D:
                heatmaps = np.zeros((68, 256, 256), dtype=np.float32)
                for i in range(68):
                    if pts[i, 0] > 0 and pts[i, 1] > 0:
                        heatmaps[i] = draw_gaussian(
                            heatmaps[i], pts[i], 2)
                heatmaps = torch.from_numpy(
                    heatmaps).unsqueeze_(0)

                heatmaps = heatmaps.to(self.device, dtype=self.dtype)
                depth_pred = self.depth_prediciton_net(
                    torch.cat((inp, heatmaps), 1)).data.cpu().view(68, 1).to(dtype=torch.float32)
                pts_img = torch.cat(
                    (pts_img, depth_pred * (1.0 / (256.0 / (200.0 * scale)))), 1)

            landmarks.append(pts_img.numpy())
            landmarks_scores.append(scores)

        if not return_bboxes:
            detected_faces = None
        if not return_landmark_score:
            landmarks_scores = None
        if return_bboxes or return_landmark_score:
            return landmarks, landmarks_scores, detected_faces
        else:
            return landmarks
