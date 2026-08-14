
"""Trains a PyTorch image classification model using device-agnostic code"""

import os
import torch
from torchvision import transforms
import data_setup, engine, model, utils
from timeit import default_timer as timer


#Setup hyperparamters
NUM_EPOCHS = 100
BATCH_SIZE = 32
HIDDEN_UNITS = 32
LEARNING_RATE = 0.001

def main():
    #Setup directories
    train_dir = "data/pizza_steak_sushi/train"
    test_dir = "data/pizza_steak_sushi/test"
    
    #Setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    #data_transforms = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()])
    train_data_transform_trivial = transforms.Compose([transforms.Resize(size=(64,64)),
                                             transforms.TrivialAugmentWide(num_magnitude_bins=31), 
                                             transforms.ToTensor()])
    #test_transform_simple = transforms.Compose([transforms.Resize(size=(64,64)),
    #                                       transforms.ToTensor()])

    
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                  test_dir=test_dir,
                                                                                  transform=train_data_transform_trivial, batch_size=BATCH_SIZE)
    #Create model
    model_2 = model.TinyVGG(input_shape=3,hidden_units=HIDDEN_UNITS, output_shape=len(class_names)).to(device)
    
    #setup loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model_2.parameters(), lr=LEARNING_RATE)
    
    #start timer
    start_time = timer()
    
    #Start training with the help of engine.py
    engine.train(model=model_2, 
                 train_dataloader=train_dataloader, 
                 test_dataloader=test_dataloader, 
                 optimizer=optimizer, 
                 loss_fn=loss_fn, 
                 epochs=NUM_EPOCHS, 
                 device=device)
    
    end_time = timer()
    print(f"Total training time:{end_time-start_time:.3f} seconds")
    
    #save the model
    utils.save_model(model=model_2, target_dir="models", model_name="5_going_modular_script_mode_tinyvgg_model.pth")
if __name__ == "__main__":
    main()