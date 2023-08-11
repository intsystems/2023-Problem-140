import torch
import torchvision
import torchvision.transforms as transforms


def get_dataloaders(dataset_name, *, batch_size: int = 64, img_size: int = 33):
    train_dl = get_dataloader(dataset_name, train=True, batch_size=batch_size, img_size=img_size)
    test_dl = get_dataloader(dataset_name, train=False, batch_size=batch_size, img_size=img_size)
    return train_dl, test_dl


def get_dataloader(dataset_name, *, train: bool, batch_size: int = 64, img_size: int = 33):
    transform = get_transform(train, img_size=img_size)
    dataset = get_dataset(dataset_name, train=train, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2)
    return dataloader


def get_dataset(dataset_name, *, train, transform):
    assert dataset_name in {'CIFAR10', 'CIFAR100', 'ImageNet'}
    if dataset_name == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    elif dataset_name == 'CIFAR100':
        dataset = torchvision.datasets.CIFAR100(root='./data', train=train, download=True, transform=transform)
    elif dataset_name == 'ImageNet':
        dataset = torchvision.datasets.ImageNet(root='./data', train=train, download=True, transform=transform)
    else:
        pass # unreachable
    return dataset


def get_transform(*, train: bool, img_size: int = 33):
    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    return transform
