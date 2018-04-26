from torch import nn
from utils.image import get_model


class FeatureExtractor(nn.Module):
    def __init__(self, arch="vgg16"):
        super().__init__()
        self.model, _ = get_model(arch)

    def forward(self, x):
        return self.model(x)
