
"""Contains functions for training and testing a PyTorch model"""
from typing import Dict, List, Tuple
import torch 
from tqdm.auto import tqdm

def train_step(model:torch.nn.Module,
              dataloader:torch.utils.data.DataLoader,
              loss_fn:torch.nn.Module,
              optimizer:torch.optim.Optimizer,
              device:torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch

    Turns a target PyTorch model to training mode and then runs through all of the
    required training steps (forward pass, loss calculation, optimizer step).

    Args:
        model:A PyTorch model to be trained.
        dataloader:A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer:A PyTorch optimizer to minimize the loss function.
        device:A target device to compute on (e.g.'cuda' or 'cpu')

    Returns:
        A tuple of training loss and trainig accuracy metrics.
        In the form (train_loss, train_accuracy), for example: 0.1112, 0.8743
        """

    model.train()
    train_loss = 0
    train_acc = 0
    #Loop through data loader data functions
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # 1. Forward pass
        y_pred = model(X)
        #2. Calculate and accumulate the loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        #3. optimizer zero grad
        optimizer.zero_grad()
        #4. Loss Backward or Backpropagation
        loss.backward()
        #5.Optimizer step
        optimizer.step()
        #Calculating and accumulating accuracy metric 
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class==y).sum().item() / len(y_pred)

    #Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model:torch.nn.Module,
               dataloader:torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               device:torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to eval mode and then performs
    a forward pass on a testing dataset.

    Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate the loss on the test data.
        device: A target device to compute on (e.g. 'cuda' or 'cpu')

    Returns:
        A tuple of testing loss amd testing accuracy metrics.
        In the form (test_loss, test_accuracy), for example: (0.0215, 0.8965)
    """
    #Putting model on eval model
    model.eval()

    test_loss, test_acc = 0, 0

    #turn on inference context manager
    with torch.inference_mode():
        #Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            #1.forward pass
            test_pred = model(X)
            #2.calculate the loss and acumulate it
            loss = loss_fn(test_pred, y)
            test_loss += loss.item()
            #Calculate and accumulate the accuracy 
            test_pred_labels = test_pred.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

    #Adjust metrics to get avarage loss and average accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(model:torch.nn.Module,
         train_dataloader:torch.utils.data.DataLoader,
         test_dataloader:torch.utils.data.DataLoader,
         optimizer:torch.optim.Optimizer,
         loss_fn:torch.nn.Module,
         epochs:int,
         device:torch.device) -> Dict[str, List[float]]:
    """Trains and tests a PyTorch Model.

    Passes a target PyTorch model through train_step() and test_step() fucntions
    for a number of epochs, training and testing the model in the same epoch loop.
    Calculate, print and stores evaluation metrics throughout.

    Args:
        model:A PyTorch model to be trained.
        train_dataloader:A DataLoader instance for the model to be trained on.
        test_dataloader:A DataLoader instance for the model to be test on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer:A PyTorch optimizer to minimize the loss function.
        epochs: An integer indicating how many times to train for.
        device:A target device to compute on (e.g.'cuda' or 'cpu')

    Returns:
        A dictionary of training and testing loss as well as trainign and testing accuracy
        metrics. Each metric has a value in a list for each epoch. 
        In the form: {train_loss:[....],
                      train_acc:[....],
                      test_loss:[....],
                      test_acc:[....]}
    """

    results = {"train_loss":[], "train_acc":[], "test_loss":[], "test_acc":[]}

    #loop through the train and test step for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn, optimizer=optimizer, device=device)
        test_loss, test_acc = test_step(model=model,
                                       dataloader=test_dataloader,
                                       loss_fn=loss_fn, device=device)
        print(f"Epoch:{epoch+1} | train_loss:{train_loss:.4f}, train_acc:{train_acc:.4f} | test_loss:{test_loss:.4f}, test_acc:{test_acc:.4f}")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results
