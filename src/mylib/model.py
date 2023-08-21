import torch
import torch.nn as nn

from torchvision.models import resnet18, ResNet18_Weights


#     def forward_hook(self, layer_name):
#         def hook(module, inp, out):
#             self.selected_output[layer_name] = out
#         return hook


class ResNet18(nn.Module):
    def __init__(self, dataset_name, *, path_to_model=None):
        super(ResNet18, self).__init__()
        self.init_for(dataset_name, path_to_model)

    def init_for(self, dataset_name, path_to_model=None):
        assert dataset_name in {'CIFAR10', 'CIFAR100', 'ImageNet'}
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        if dataset_name != 'ImageNet':
            assert path_to_model != None
            num_classes = {'CIFAR10':10, 'CIFAR100':100}
            self.model.fc = nn.Linear(512, num_classes[dataset_name])
            self.load_state_dict(torch.load(path_to_model))
        # self.fhooks = []
        # for layer in self.model._modules.keys():
        #     self.fhooks.append(getattr(self.model, layer).register_forward_hook(self.forward_hook(layer)))

    def forward(self, x):
        return self.model(x)

