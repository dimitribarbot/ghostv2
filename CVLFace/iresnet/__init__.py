import torch

from ArcFace.iresnet import IResNet
from CVLFace.utils import get_parameter_device, get_parameter_dtype


class IResNetModel(torch.nn.Module):

    """
    A class representing a model for IResNet architectures. It supports creating
    models with specific configurations such as IR_50 and IR_101.
    Attributes:
        net (torch.nn.Module): The IResNet network (either IR_50 or IR_101).
        config (object): The configuration object with model specifications.
    """


    def __init__(self, net: IResNet):
        super(IResNetModel, self).__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)

    @property
    def device(self) -> torch.device:
        """
        Returns the device of the model's parameters.
        Returns:
            device: The device the model is on.
        """
        return get_parameter_device(self)

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns the data type of the model's parameters.
        Returns:
            torch.dtype: The data type of the model.
        """
        return get_parameter_dtype(self)

    def num_parameters(self, only_trainable: bool = False) -> int:
        """
        Returns the number of parameters in the model, optionally filtering only trainable parameters.
        Parameters:
            only_trainable (bool, optional): Whether to count only trainable parameters. Default is False.
        Returns:
            int: The number of parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not only_trainable)

    def has_trainable_params(self):
        """
        Checks if the model has any trainable parameters.
        Returns:
            bool: True if the model has trainable parameters, False otherwise.
        """
        return any(p.requires_grad for p in self.parameters())