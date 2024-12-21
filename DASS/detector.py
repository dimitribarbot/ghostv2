import os

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from safetensors.torch import load_file

from DASS.models.CFA import CFA
from DASS.models.yolox import YOLOX
from DASS.models.yolo_head import YOLOXHead
from DASS.models.yolo_head_stem import YOLOXHeadStem
from DASS.models.yolo_pafpn import YOLOPAFPN
from DASS.utils import postprocess, preprocess


def make_real_path(relative_path):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), relative_path)


def flatten(l):
    return [item for sublist in l for item in sublist]


def chunk_generator(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


class DassDetFace:
    def __init__(
        self,
        gpu_id=-1,
        model_path=None,
        landmark_detector_model_path=None,
        model_mode=1,
        device = "cuda",
        fp16: bool = False,
        depth: float = 1.33,
        width: float = 1.25,
        num_landmark: int = 24,
    ):
        self.gpu_id = gpu_id if device != "mps" else 0
        self.device = (
            torch.device("cpu") if gpu_id == -1 else torch.device(device, gpu_id)
        )
        # self.model = YOLOX(
        #     backbone=YOLOPAFPN(depth=depth, width=width),
        #     head_stem=YOLOXHeadStem(width=width),
        #     face_head=YOLOXHead(1, width=width),
        #     body_head=YOLOXHead(1, width=width)
        # )
        # self.model.load_state_dict(load_file(model_path), strict=True)
        # self.model.eval()
        # self.model = self.model.to(device)
        # self.model_mode = model_mode
        self.fp16 = fp16
        # if self.fp16:
        #     self.model.half()
        self.face_detector = cv2.CascadeClassifier(make_real_path("./models/lbpcascade_animeface.xml"))
        self.landmark_detector = CFA(output_channel_num=num_landmark + 1)
        self.landmark_detector.load_state_dict(load_file(landmark_detector_model_path), strict=True)
        self.landmark_detector.eval()
        self.landmark_detector = self.landmark_detector.to(device)
        self.num_landmark = num_landmark

    def convert_24_landmarks_to_5(self, landmarks24: list[np.ndarray]):
        lm_idx = np.array([10, 15, 20, 21, 23], dtype=np.int32) - 1
        landmarks5 = np.stack([
            landmarks24[lm_idx[1], :],  # left eye
            landmarks24[lm_idx[2], :],  # right eye
            landmarks24[lm_idx[0], :],  # nose
            landmarks24[lm_idx[3], :],  # lip
            landmarks24[lm_idx[4], :]   # lip
        ], axis=0)
        return landmarks5
    
    def get_optimal_size_from_face(self, face, oiginal_image_width):
        x_, y_, w_, h_ = face
        x = max(x_ - w_ / 8, 0)
        rx = min(x_ + w_ * 9 / 8, oiginal_image_width)
        y = max(y_ - h_ / 4, 0)
        by = y_ + h_
        w = rx - x
        h = by - y
        return x, y, w, h
    
    def get_bbox_from_face(self, face, oiginal_image_width):
        x, y, w, h = self.get_optimal_size_from_face(face, oiginal_image_width)
        return np.array([x, y, x + w, y + h])

    @torch.no_grad()
    def batch_detect(self, images, device, is_tensor=False, threshold=0.5, cv=False, return_dict=False, fp16=False):
        nms_thold  = 0.65
        conf_thold = threshold

        if fp16:
            dtype = torch.float16
        else:
            dtype = torch.float32

        if is_tensor:
            images = images.cpu().detach().numpy()
        else:
            images = np.array(images)

        if cv:
            images = images[:, :, :, [2, 1, 0]]  # bgr to rgb

        all_dets = []
        for image in images:
            detected_faces = self.face_detector.detectMultiScale(image[:, :, ::-1])

            if len(detected_faces) > 0:
                # transform image
                imgs_tmp = []
                for face in detected_faces:
                    x, y, w, h = self.get_optimal_size_from_face(face, image.shape[1])
                    img_tmp = image[int(y):int(y + h), int(x):int(x + w)]
                    img_tmp = cv2.resize(img_tmp, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
                    imgs_tmp.append(img_tmp)

                imgs_tmp = np.array(imgs_tmp).transpose((0, 3, 1, 2))
                imgs_tmp = torch.from_numpy(imgs_tmp).contiguous().to(device=device, dtype=torch.float32).div(255)
                denorm_images = F.normalize(imgs_tmp, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], False)

                heatmaps = self.landmark_detector(denorm_images)
                heatmaps = heatmaps[-1].cpu().detach().numpy()

                all_landmarks24 = []
                for face, heatmap in zip(detected_faces, heatmaps):
                    x, y, w, h = self.get_optimal_size_from_face(face, image.shape[1])

                    landmarks24 = []
                    for i in range(self.num_landmark):
                        heatmap_tmp = cv2.resize(heatmap[i], (128, 128), interpolation=cv2.INTER_CUBIC)
                        landmark = np.unravel_index(np.argmax(heatmap_tmp), heatmap_tmp.shape)
                        landmark_y = landmark[0] * h / 128
                        landmark_x = landmark[1] * w / 128
                        landmarks24.append([x + landmark_x, y + landmark_y])
                    all_landmarks24.append(landmarks24)
                        
                all_landmarks5 = [self.convert_24_landmarks_to_5(np.array(landmarks24)) for landmarks24 in all_landmarks24]

                faces = [(self.get_bbox_from_face(face, image.shape[1]), None, landmarks5, landmarks24) for (face, landmarks5, landmarks24) 
                         in zip(detected_faces, all_landmarks5, all_landmarks24)]
                
                all_dets.append(faces)

        if len(all_dets) == 0:
            all_dets.append([(None, None, None, None)])


        # preprocessed_images = []
        # img_sizes = [[], []]
        # img_resizes = [[], []]
        # for image in images:
        #     height, width, _ = image.shape
        #     resize_size = ((height // 32) * 32, (width // 32) * 32)
        #     preprocessed_image = preprocess(image, resize_size)
        #     img_sizes[0].append(height)
        #     img_sizes[1].append(width)
        #     img_resizes[0].append(resize_size[0])
        #     img_resizes[1].append(resize_size[1])
        #     preprocessed_images.append(preprocessed_image)

        # preprocessed_images = torch.from_numpy(np.array(preprocessed_images)).to(device=device, dtype=dtype)

        # outputs, _ = self.model(preprocessed_images, mode=self.model_mode)
        # outputs = postprocess(outputs, conf_thold, nms_thold)

        # all_dets = []
        # if len(outputs) > 0:
        #     for (image, output_img, img_h, img_w, img_resized_h, img_resized_w) in zip(
        #         images, outputs, img_sizes[0], img_sizes[1], img_resizes[0], img_resizes[1]
        #     ):
        #         if output_img is None:
        #             faces = [(None, None, None, None)]
        #         else:
        #             output_img = output_img.cpu().detach().numpy()

        #             scale = min(img_resized_h / float(img_h), img_resized_w / float(img_w))
                    
        #             bboxes = [output[0:4] / scale for output in output_img]
        #             scores = [output[4] for output in output_img]

        #             detected_faces = self.face_detector.detectMultiScale(image[:, :, ::-1])
        #             if len(detected_faces) == 0:
        #                 faces = [(bbox, score, None, None) for (bbox, score) in zip(bboxes, scores)]
        #             else:
        #                 # transform image
        #                 imgs_tmp = []
        #                 for bbox in bboxes:
        #                     x = max(bbox[0] - (bbox[2] - bbox[0]) / 8, 0)
        #                     rx = min(bbox[0] + (bbox[2] - bbox[0]) * 9 / 8, image.shape[1])
        #                     y = max(bbox[1] - (bbox[3] - bbox[1]) / 4, 0)
        #                     by = bbox[1] + (bbox[3] - bbox[1])
        #                     w = rx - x
        #                     h = by - y
        #                     img_tmp = image[int(y):int(y + h), int(x):int(x + w)]
        #                     img_tmp = cv2.resize(img_tmp, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
        #                     imgs_tmp.append(img_tmp)

        #                 imgs_tmp = np.array(imgs_tmp).transpose((0, 3, 1, 2))
        #                 imgs_tmp = torch.from_numpy(imgs_tmp).contiguous().to(device=device, dtype=torch.float32).div(255)
        #                 denorm_images = F.normalize(imgs_tmp, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], False)

        #                 heatmaps = self.landmark_detector(denorm_images)
        #                 heatmaps = heatmaps[-1].cpu().detach().numpy()

        #                 all_landmarks24 = []
        #                 for (bbox), heatmap in zip(bboxes, heatmaps):
        #                     x = max(bbox[0] - (bbox[2] - bbox[0]) / 8, 0)
        #                     rx = min(bbox[0] + (bbox[2] - bbox[0]) * 9 / 8, image.shape[1])
        #                     y = max(bbox[1] - (bbox[3] - bbox[1]) / 4, 0)
        #                     by = bbox[1] + (bbox[3] - bbox[1])
        #                     w = rx - x
        #                     h = by - y

        #                     landmarks24 = []
        #                     for i in range(self.num_landmark):
        #                         heatmap_tmp = cv2.resize(heatmap[i], (128, 128), interpolation=cv2.INTER_CUBIC)
        #                         landmark = np.unravel_index(np.argmax(heatmap_tmp), heatmap_tmp.shape)
        #                         landmark_y = landmark[0] * h / 128
        #                         landmark_x = landmark[1] * w / 128
        #                         landmarks24.append([x + landmark_x, y + landmark_y])
        #                     all_landmarks24.append(landmarks24)
                                
        #                 all_landmarks5 = [self.convert_24_landmarks_to_5(np.array(landmarks24)) for landmarks24 in all_landmarks24]

        #                 faces = [(bbox, score, landmarks5, landmarks24) for (bbox, score, landmarks5, landmarks24) 
        #                         in zip(bboxes, scores, all_landmarks5, all_landmarks24)]

        #         all_dets.append(faces)
        # else:
        #     all_dets.append([(None, None, None, None)])

        if return_dict:
            all_dict_results = []
            for faces in all_dets:
                dict_results = []
                for face in faces:
                    box, score, landmark5, landmark24 = face
                    dict_results.append(
                        {
                            "box": box,
                            "kps": landmark5,
                            "kps24": landmark24,
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