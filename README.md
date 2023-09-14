# Animals and Vehicles Multi-Class Classifer

This is my first attempt at making my own image classifier. I am using the ResNet18 model and the Cifar10 dataset from HuggingFace, to create a program that can classify 10 different classes of animals and vehicles. 

## Preparing the data
The dataset provided by HuggingFace already has separate training and validation sets. After loading the data, I am splitting the data into these sets and mapping the Image data into tensors and apply making random modifications (e.g. rotations) to the data using torchvision.transforms before feeding it into into PyTorch DataLoaders. These random modifications are meant to help increase the model accuracy by creating more varied and random image data.

The image data has the features: "img" and "label" which I am treating as the model input and prediction respectively.

The training set DataLoader will fead data in batch sizes of 32. I have used this batch size to avoid overfitting the model to the data or get the model gradient optimizer to be attracted to saddle points, that may occur with larger batch sizes.

## Training the model

The ResNet18 model that I have loaded has already been pretrained on large datasets.

I have chosen a Batch Size of 32, a Cyclical Learning Rate with base_lr = 0.001 and max_lr=0.01, a Momentum of 0.9, and 10 Epochs (for model training).

The Cyclical Learning Rate was done using the lr_scheduler.CyclicLR from torch.optim, and was used to remove the need to find an optimal learning rate in a way that is less computationally expensive as the Adaptive Learning Rate method.

I am using the Stochastic Gradient Descent method as my Gradient Optimizer to train my model parameters with the Cyclical Learning Rate and Momentum. I am using SGD since regular Gradient Descent is more computationally expensive and leads to slower convergence to the global minimum of the gradient, even if it may result in higher accuracy. The purpose of Momentum is to increase performance and efficiency by allowing the algorithm to build inertia in order to avoid oscillations in messy gradients and quickly pass over smooth areas.

I am using Cross Entropy Loss as my Loss Criterion, as it is better suited for multi-class classification problems since it uses the SoftMax activation function. The SoftMax activation function takes in data from the output layer of the model, and converts it into a vector with corresponding probabilities. The highest probability is the model prediction. I am using Cross Entropy Loss as opposed to BCELoss, which is better for binary classification.

I chose 10 epochs, since I wanted to avoid overtraining the model on the training set.

I have a separate function which trains the model to the Cifar10 data called train_model. This function takes in the model, training and validation DataLoaders, Loss Criterion, Gradient Optimizer, and # of Epochs. The function trains the data according to the number of epochs and the training batch size. For each epoch, it makes predictions for the validation set, which is used to record the cost and accuracy of the model as it is being trained. It uses tqdm and print statements to output relevant information such as learning rate, validation cost, validation accuracy, and epoch # for each run of the training data and show progress.

## Cost v Accuracy Graph

The train_model function also outputs the cost and accuracy for each epoch as a list. This is then graphed as a Cost v Accuracy figure to show how the progress of the model predictions as it is trained to the data.
