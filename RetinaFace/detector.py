# Code adapted from https://github.com/elliottzheng/batch-face/blob/master/batch_face/face_detection/detector.py

import numpy as np
import torch

from RetinaFace.alignment import load_net, batch_detect


def flatten(l):
    return [item for sublist in l for item in sublist]


def chunk_generator(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


class RetinaFace:
    def __init__(
        self,
        gpu_id=-1,
        model_path=None,
        device = "cuda",
        return_dict: bool = False,
        fp16: bool = False,
    ):
        self.gpu_id = gpu_id if device != "mps" else 0
        self.device = (
            torch.device("cpu") if gpu_id == -1 else torch.device(device, gpu_id)
        )
        self.model = load_net(model_path, self.device)
        self.fp16 = fp16
        if self.fp16:
            self.model.half()
        self.return_dict = return_dict
    
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
                return batch_detect(self.model, [images], **kwargs)[0]
            elif len(images.shape) == 4:
                return batch_detect(self.model, images, **kwargs)
        elif isinstance(images, list):
            return batch_detect(self.model, np.array(images), **kwargs)
        elif isinstance(images, torch.Tensor):
            kwargs["is_tensor"] = True
            if len(images.shape) == 3:
                return batch_detect(self.model, images.unsqueeze(0), **kwargs)[0]
            elif len(images.shape) == 4:
                return batch_detect(self.model, images, **kwargs)
        else:
            raise NotImplementedError(f"images type {type(images)} not supported")

    def pseudo_batch_detect(self, images, **kwargs):
        assert "chunk_size" not in kwargs
        return [self.detect(image, **kwargs) for image in images]

    def __call__(self, images, **kwargs):
        return self.detect(images, **kwargs)