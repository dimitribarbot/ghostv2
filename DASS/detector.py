# Code adapted from https://github.com/elliottzheng/batch-face/blob/master/batch_face/face_detection/detector.py

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from DASS.models.yolox import YOLOX
from DASS.models.yolo_head import YOLOXHead
from DASS.models.yolo_head_stem import YOLOXHeadStem
from DASS.models.yolo_pafpn import YOLOPAFPN
from DASS.utils import postprocess, preprocess
from FaceAlignment.api import FaceAlignment, LandmarksType


def flatten(l):
    return [item for sublist in l for item in sublist]


def chunk_generator(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


class DassDetFace:
    def __init__(
        self,
        face_alignment: FaceAlignment,
        gpu_id=-1,
        model_path=None,
        model_mode=1,
        device = "cuda",
        fp16: bool = False,
        depth: float = 1.33,
        width: float = 1.25
    ):
        self.gpu_id = gpu_id if device != "mps" else 0
        self.device = (
            torch.device("cpu") if gpu_id == -1 else torch.device(device, gpu_id)
        )
        self.model = YOLOX(backbone=YOLOPAFPN(depth=depth, width=width),
              head_stem=YOLOXHeadStem(width=width),
              face_head=YOLOXHead(1, width=width),
              body_head=YOLOXHead(1, width=width))
        self.model.load_state_dict(load_file(model_path), strict=True)
        self.model.eval()
        self.model = self.model.to(device)
        self.model_mode = model_mode
        self.fp16 = fp16
        if self.fp16:
            self.model.half()
        self.face_alignment = face_alignment

    def convert_68_landmarks_to_5(self, landmarks68: list[np.ndarray]):
        lm_idx = np.array([31, 37, 40, 43, 46, 49, 55], dtype=np.int32) - 1
        landmarks5 = np.stack([
            np.mean(landmarks68[lm_idx[[1, 2]], :], 0),  # left eye
            np.mean(landmarks68[lm_idx[[3, 4]], :], 0),  # right eye
            landmarks68[lm_idx[0], :],  # nose
            landmarks68[lm_idx[5], :],  # lip
            landmarks68[lm_idx[6], :]   # lip
        ], axis=0)
        return landmarks5

    @torch.no_grad()
    def batch_detect(self, images, device, is_tensor=False, threshold=0.5, cv=False, return_dict=False,  fp16=False):
        nms_thold  = 0.65
        conf_thold = threshold

        if fp16:
            dtype = torch.float16
        else:
            dtype = torch.float32

        if is_tensor:
            images = images.cpu().numpy()
        else:
            images = np.array(images)

        if cv:
            images = images[:, :, :, [2, 1, 0]]  # rgb to bgr

        preprocessed_images = []
        img_sizes = [[], []]
        img_resizes = [[], []]
        for image in images:
            height, width, _ = image.shape
            resize_size = ((height // 32) * 32, (width // 32) * 32)
            preprocessed_image = preprocess(image, resize_size)
            img_sizes[0].append(height)
            img_sizes[1].append(width)
            img_resizes[0].append(resize_size[0])
            img_resizes[1].append(resize_size[1])
            preprocessed_images.append(preprocessed_image)

        preprocessed_images = torch.from_numpy(np.array(preprocessed_images)).to(device=device, dtype=dtype)

        outputs, _ = self.model(preprocessed_images, mode=self.model_mode)
        outputs = postprocess(outputs, conf_thold, nms_thold)

        all_dets = []
        if len(outputs) > 0:
            for (image, output_img, img_h, img_w, img_resized_h, img_resized_w) in zip(
                images, outputs, img_sizes[0], img_sizes[1], img_resizes[0], img_resizes[1]
            ):
                if output_img is None:
                    faces = []
                else:
                    output_img = output_img.cpu().numpy()

                    scale = min(img_resized_h / float(img_h), img_resized_w / float(img_w))
                    
                    bboxes = [output[0:4] / scale for output in output_img]
                    scores = [output[4] for output in output_img]

                    landmarks68 = self.face_alignment.get_landmarks_from_image(
                        image,
                        detected_faces=bboxes,
                    )
                    landmarks5 = [self.convert_68_landmarks_to_5(landmark68) for landmark68 in landmarks68]

                    faces = [(bbox, score, landmark5, landmark68) for (bbox, score, landmark5, landmark68) 
                            in zip(bboxes, scores, landmarks5, landmarks68)]

                all_dets.append(faces)

        if return_dict:
            all_dict_results = []
            for faces in all_dets:
                dict_results = []
                for face in faces:
                    box, score, landmarks5, landmarks68 = face
                    dict_results.append(
                        {
                            "box": box,
                            "kps": landmarks5,
                            "kps68": landmarks68,
                            "score": score,
                        }
                    )
                all_dict_results.append(dict_results)
            return all_dict_results
        else:
            return all_dets

    
    @torch.inference_mode()
    def detect(self, images, chunk_size=None, batch_size=None, **kwargs):
        """
        cv: True if is bgr
        chunk_size: batch size
        """
        if self.fp16:
            kwargs["fp16"] = True
        # do not specify chunk_size and batch_size at the same time
        assert not (chunk_size is not None and batch_size is not None), "chunk_size and batch_size cannot be specified at the same time, they are the same thing."

        if chunk_size is not None:
            batch_size = chunk_size

        if batch_size is not None:    
            return flatten([self.detect(part, **kwargs) for part in chunk_generator(images, batch_size)])

        kwargs["device"] = self.device
        if isinstance(images, np.ndarray):
            if len(images.shape) == 3:
                return self.batch_detect([images], **kwargs)[0]
            elif len(images.shape) == 4:
                return self.batch_detect(images, **kwargs)
        elif isinstance(images, list):
            return self.batch_detect(np.array(images), **kwargs)
        elif isinstance(images, torch.Tensor):
            kwargs["is_tensor"] = True
            if len(images.shape) == 3:
                return self.batch_detect(images.unsqueeze(0), **kwargs)[0]
            elif len(images.shape) == 4:
                return self.batch_detect(images, **kwargs)
        else:
            raise NotImplementedError(f"images type {type(images)} not supported")

    def pseudo_batch_detect(self, images, **kwargs):
        assert "chunk_size" not in kwargs
        return [self.detect(image, **kwargs) for image in images]

    def __call__(self, images, **kwargs):
        return self.detect(images, **kwargs)