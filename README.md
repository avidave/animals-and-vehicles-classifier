# Animals and Vehicles Multi-Class Classifer

This is my first attempt at making my own image classifier. I am using the ResNet18 model and the Cifar10 dataset from HuggingFace, to create a program that can classify 10 different classes of animals and vehicles. 

## Preparing the data
The dataset provided by HuggingFace already has separate training and validation sets. After loading the data, I am splitting the data into these sets and mapping the Image data into tensors and apply making random modifications (e.g. rotations) to the data using torchvision.transforms before feeding it into into PyTorch DataLoaders. These random modifications are meant to help increase the model accuracy by creating more varied and random image data.

The image data has the features: "img" and "label" which I am treating as the model input and prediction respectively.

The training set DataLoader will fead data in batch sizes of 32. I have used this batch size to avoid overfitting the model to the data or get the model gradient optimizer to be attracted to saddle points, that may occur with larger batch sizes.

## Training the model

The ResNet18 model that I have loaded has already been pretrained on large datasets.

I have chosen a batch size of 32, a cyclical learning rate with base_lr = 0.00001 and max_lr=0.001, 

I have a separate function which trains the model to the Cifar10 data. 
