import torch
from torchvision import models


def get_model(arch):
    if arch == 'vgg':
        model = models.vgg16(pretrained=True)
        model.features = torch.nn.DataParallel(model.features)
        modules = list(model.classifier.children())
        # restrict to the FC layer that gives us the 4096 embedding
        modules = modules[:-3]
        model.classifier = torch.nn.Sequential(*modules)
        layer = "fc7"

    elif arch == 'resnet152':
        model = models.resnet152(pretrained=True)
        modules = list(model.children())[:-2]
        model = torch.nn.Sequential(*modules)
        layer = "pool5"

    return model, layer


def coco_name_format(image_id, split):
    image_name = "COCO_{0}2014_{1:012}.jpg".format(split, image_id)
    return image_name
