import torch
import torch.nn as nn

from torchvision.models import resnet18, ResNet18_Weights


class ResNet18(nn.Module):
    def __init__(self, dataset_name, *, path_to_model=None):
        super(ResNet18, self).__init__()
        self.init_for(dataset_name, path_to_model)

    def init_for(self, dataset_name, path_to_model=None):
        assert dataset_name in {'CIFAR10', 'CIFAR100', 'ImageNet'}
        if dataset_name == 'ImageNet':
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            assert path_to_model != None
            self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
            num_classes = 10 if dataset_name == 'CIFAR10' else 100
            self.model.fc = nn.Linear(512, num_classes)
            self.model.load_state_dict(torch.load(path_to_model))

    def forward(self, x):
        return self.model(x)

