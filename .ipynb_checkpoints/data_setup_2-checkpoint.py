
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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
