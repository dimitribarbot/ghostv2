from ..core import FaceDetector


class DefaultDetector(FaceDetector):
    '''Default Detector.
    '''

    def __init__(self, device, path_to_detector=None, verbose=False, threshold=0.5, fp16=True):
        super(DefaultDetector, self).__init__(device, verbose)

    def detect_from_image(self, tensor_or_path):
        return []

    def detect_from_batch(self, tensor):
        return []

    @property
    def reference_scale(self):
        return 195

    @property
    def reference_x_shift(self):
        return 0

    @property
    def reference_y_shift(self):
        return 0