# Abstract Art Generation using DCGAN
![generated-images-0200](https://user-images.githubusercontent.com/70977847/210127549-c6d23252-f000-4d8a-8f22-d5ff2a1718fa.png)
### The discriminator
It takes an image as input, and tries to classify it as "real" or "generated". In this sense, it's like any other neural network. We'll use a convolutional neural networks (CNN) which outputs a single number output for every image. We'll use stride of 2 to progressively reduce the size of the output feature map.
### The generator 
The input to the generator is typically a vector or a matrix of random numbers (referred to as a latent tensor) which is used as a seed for generating an image. The generator will convert a latent tensor of shape (128, 1, 1) into an image tensor of shape 3 x 128 x 128. To achive this, we'll use the ConvTranspose2d layer from PyTorch, which is performs to as a transposed convolution (also referred to as a deconvolution)
### Discriminator training 
![](https://camo.githubusercontent.com/357431aaed2ade4cc892c62b0b87b72840ad5d90bf8f2e5bde1cae62afaf8df6/68747470733a2f2f692e696d6775722e636f6d2f364e4d644f39752e706e67)
<br>
Since the discriminator is a binary classification model, we can use the binary cross entropy loss function to quantify how well it is able to differentiate between real and generated images.
<br>
Here are the steps involved in training the discriminator.

* We expect the discriminator to output 1 if the image was picked from the real MNIST dataset, and 0 if it was generated using the generator network.
* We first pass a batch of real images, and compute the loss, setting the target labels to 1.
* Then we pass a batch of fake images (generated using the generator) pass them into the discriminator, and compute the loss, setting the target labels to 0.
* Finally we add the two losses and use the overall loss to perform gradient descent to adjust the weights of the discriminator.
* It's important to note that we don't change the weights of the generator model while training the discriminator (opt_d only affects the discriminator.parameters())
# Major drawbacks 
Due to smalll size of training set (~ 3000 images) our model couldn't be trained effectively
