# cats_and_dogs
 This is a binary classification model based on CNN. The main methods to change the parameters and improve the model performance are as follows. The current accuracy  is 82%

1. Increase the batch_size to reduce the number of gradient updates and speed up the training, but it also increases the memory usage.

2. Modify the learning rate (lr) parameter, which can help the model to converge better by gradually decreasing the learning rate.

3. Add more convolutional layers or fully connected layers to increase the depth and complexity of the model to improve its learning ability.

4. Adjust the parameters in ImageDataGenerator, such as changing the rotation angle, scaling range, etc., to increase the diversity of the data so that the model can generalize better to new data.

5. Try different optimizers, such as Adam, RMSprop, etc., and tune their hyperparameters.

6. Increase the training cycles (epochs)
