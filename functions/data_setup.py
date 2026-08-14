"""Contains functionality for creating PyTorch DataLoader's for
image classification data.
"""

import os

from torchvision import datasets,  transforms
from torch.utils.data import DataLoader

NUM_WORKERS = 0

def create_dataloaders(train_dir:str,
                      test_dir:str,
                      transform:transforms.Compose,
                      batch_size:int,
                      num_workers:int=NUM_WORKERS):
    """Creates training and testing dataloaders.

    Takes in testing and training directory path and turns them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
        train_dir: Path to training directory.
        test_dir: Path to testing directory.
        transform: torchvision transforms to perform on training and testing data.
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: An ineteger for number of workers per DataLoader.

    Returns:
        A tuple of (train_datalloader, test_dataloader, classnames).
        where class_names is a list of target classes.
        Example usage:
            train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=path/to/train_dir,
                test_dir=apth/to/test_dir,
                transform=some_transform,
                batch_size=32, num_workers=12)"""

    train_data = datasets.ImageFolder(root=train_dir,
                                 transform=transform)
    test_data = datasets.ImageFolder(root=test_dir,
                                transform=transform)
    #Get the class names
    class_names = train_data.classes

    #Turn our images to DataLoader
    train_dataloader = DataLoader(train_data,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 pin_memory=True)

    test_dataloader = DataLoader(test_data,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True)

    return train_dataloader, test_dataloader, class_names


def create_food101_dataloaders(
    root_dir: str,
    train_transform: transforms.Compose,
    test_transform: transforms.Compose,
    batch_size: int,
    num_workers: int = 0
):
    """
    Creates training and testing DataLoaders for Food-101.

    Args:
        root_dir: Root directory containing the food-101 folder.
                  Example: "data"
        train_transform: Transformations for training images.
        test_transform: Transformations for testing images.
        batch_size: Number of samples per batch.
        num_workers: Number of DataLoader workers.

    Returns:
        train_dataloader, test_dataloader, class_names
    """

    # Create Food-101 training dataset
    train_data = datasets.Food101(
        root=root_dir,
        split="train",
        transform=train_transform,
        download=False
    )

    # Create Food-101 testing dataset
    test_data = datasets.Food101(
        root=root_dir,
        split="test",
        transform=test_transform,
        download=False
    )

    # Get class names
    class_names = train_data.classes

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader, test_dataloader, class_names
