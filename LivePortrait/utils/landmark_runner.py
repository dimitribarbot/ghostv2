import cv2
import numpy as np

from builtins import getattr, set
import torch
from torch.nn.modules import Module
from torch.nn.modules.linear import Linear
from torch.nn.modules.conv import Conv2d
from torch.fx.graph_module import reduce_graph_module
from torch._C import _VariableFunctionsClass
from onnx2torch.node_converters.identity import OnnxCopyIdentity
from onnx2torch.node_converters.reduce import OnnxReduceStaticAxes
from onnx2torch.node_converters.binary_math_operations import OnnxBinaryMathOperation, _onnx_div
from onnx2torch.node_converters.constant import OnnxConstant
from onnx2torch.node_converters.pow import OnnxPow, OnnxSqrt
from onnx2torch.node_converters.transpose import OnnxTranspose
from onnx2torch.node_converters.matmul import OnnxMatMul
from onnx2torch.node_converters.activations import OnnxErf
from onnx2torch.node_converters.reshape import OnnxReshape
from onnx2torch.node_converters.concat import OnnxConcat

from LivePortrait.utils.crop import crop_image, _transform_pts


torch.serialization.add_safe_globals([
    reduce_graph_module,
    getattr,
    _onnx_div,
    set,
    Module,
    Linear,
    Conv2d,
    OnnxCopyIdentity,
    OnnxReduceStaticAxes,
    OnnxBinaryMathOperation,
    OnnxConstant,
    OnnxPow,
    OnnxSqrt,
    OnnxTranspose,
    OnnxMatMul,
    OnnxErf,
    OnnxReshape,
    OnnxConcat,
    _VariableFunctionsClass
])

def to_ndarray(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy()
    elif isinstance(obj, np.ndarray):
        return obj
    else:
        return np.array(obj)

class LandmarkRunner(object):
    """landmark runner torch version"""
    def __init__(self, **kwargs):
        self.device = kwargs.get('device')
        self.dsize = kwargs.get('dsize', 224)
        ckpt_path = kwargs.get('ckpt_path')

        self.model = torch.load(ckpt_path, weights_only=True).to(self.device)

    def _run(self, inp):
        input = torch.from_numpy(inp).to(self.device)
        out = self.model(input)
        return out

    def run(self, img_rgb: np.ndarray, lmk=None):
        if lmk is not None:
            crop_dct = crop_image(img_rgb, lmk, dsize=self.dsize, scale=1.5, vy_ratio=-0.1)
            img_crop_rgb = crop_dct["img_crop"]
        else:
            img_crop_rgb = cv2.resize(img_rgb, (self.dsize, self.dsize))
            scale = max(img_rgb.shape[:2]) / self.dsize
            crop_dct = {
                'M_c2o': np.array([
                    [scale, 0., 0.],
                    [0., scale, 0.],
                    [0., 0., 1.],
                ], dtype=np.float32),
            }

        inp = (img_crop_rgb.astype(np.float32) / 255.).transpose(2, 0, 1)[None, ...]  # HxWx3 (BGR) -> 1x3xHxW (RGB!)

        out_lst = self._run(inp)
        out_pts = out_lst[2]

        # 2d landmarks 203 points
        lmk = to_ndarray(out_pts[0]).reshape(-1, 2) * self.dsize  # scale to 0-224
        lmk = _transform_pts(lmk, M=crop_dct['M_c2o'])

        return lmk