# Introduction to CNNs

Convolutional neural networks (CNNs) are one of the largest revolutions in the computer vision field. They are the state of the art (SOTA) solution for problems like object detection, handwriting recognition, face recognition, and many other image processing tasks.

Traditional methods are still used for specific tasks and as preprocessing steps for CNNs, so don't forget everything you've learned so far. :)

You already used convolutions many times in these labs. In lab2 you wrote your own convolution code. In lab3 you used median filtering, a convolution with a specific kernel to blur the image. In lab4 you used a blur filter.

Depending on the used kernel, convolutions can achieve all kinds of different things: Show specific features like edges and corners, show and remove textures, etc.

Traditionally, when someone is working on an image processing task they would hand-select the kernel used in the convolution.

CNNs work similarly but take it all one step further. Instead of manually convolving the image with known kernels, in deep learning we let the network learn the kernels it's using. CNNs have multiple layers, in each layer, they perform lots and lots of convolutions with different kernels in parallel. Each layer's output is used as input for the next layer. By doing this, a CNN can learn more-and-more complex features deeper in the network. At first, it can detect lines and corners, in the later layers, it can combine lines to detect more complex shapes, and so on.

Follow along this notebook to find out how CNNs work.

## PyTorch and neural networks

For this lab, you'll use PyTorch, one of the most popular deep learning libraries to implement a convolutional neural network from scratch. For simple problems, you can use tools like [fastai](https://docs.fast.ai) which can automate a lot of this process for you. Still, it's good to be familiar with what lies beneath tools like that.

# Preparing our data

To train a neural network, we first need a lot of data. Neural networks learn by looking at example input images and their correct output. In your case that will be a bunch of handwritten digits and their correct translation into text.

You can look at a neural network as a math function, mapping inputs into outputs. By knowing the correct solution during training, a neural network can adjust what it does inside the function so that its output matches the correct solution.

For this problem, you'll use a popular dataset of handwritten digits called MNIST. Torchvision already includes a function to download the dataset.

You'll apply a transformation to each image. First, you'll convert the images to PyTorch Tensors. Tensors are _very_ similar to NumPy arrays, so you're already fairly familiar with them.

The other transformation you'll apply is normalization. Since neural networks work by multiplying and adding together a bunch of numbers, the absolute values of those numbers matter a lot. Multiplying 0.001 a bunch of times will give you a very small number at the end, making it very hard to work with. Similarly, multiplying huge numbers can be unstable too. It's best to map CNN inputs into a range of [-0.5, 0.5], making sure they're centered around 0. That's what the Normalize transform does.

You'll create two datasets, one for training and one for testing. It's very important that you do this step: You can't verify that your model works if you train and test the model on the same data, since it can in some cases learn to reproduce correct outputs on the training data by memorization, not understanding.

More information about specific code and tasks are in corresponding Jupyter notebook. 

# CNN architecture

Now it's time to build the network. A CNN usually consists of convolutional layers, i.e. NN layers that perform a convolution on the input matrix and return the convolved matrix as output. You'll use two of these layers. You'll then use a dropout layer, which just randomly disables half of the features to prevent the network from memorizing the training dataset. Finally, you'll use two layers to combine all of the convolution results into just 10 numbers, the probability for each digit. Exactly how those layers are combined (whether they are summed, how much each layer contributes to the result for each digit, etc.) will be learned during training.

More information about specific code and tasks are in corresponding Jupyter notebook. 

# Single and multi-label classification with ResNets

## Introduction 

In this lab, we will build on top of the CNNs introduced in the previous (lab7) and explain to you the ResNet (residual network) architecture. Although CNNs perform well, the limited amount of annotated data requires the development of efficient and more complex, deeper network architectures. Deep convolutional neural networks have shown a significant increase in the accuracy for various segmentation and classification tasks. 

ResNet was introduced in 2015 by Kaiming He et al. in the article ["Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385) and is by far the most used model architecture nowadays. More recent developments in image models almost always use the same trick of residual connections, and most of the time, they are just a tweak of the original ResNet.

We will first show you the basic ResNet as it was first designed, then explain to you what modern tweaks make it more performant. 

### Deep Residual Networks (ResNets)

After the first CNN-based architecture (AlexNet) that win the ImageNet 2012 competition, every subsequent winning architecture uses more layers in a deep neural network to reduce the error rate. This works for less number of layers, but when we increase the number of layers, there is a common problem in deep learning associated with that called Vanishing/Exploding gradient. This causes the gradient to become 0 or too large. Thus when we increases number of layers, the training and test error rate also increases.

<p align="center">
  <img src="https://i.ibb.co/0n06TVs/resnet.png">
</p>


In the above plot, we can observe that a 56-layer CNN gives more error rate on both training and testing dataset than a 20-layer CNN architecture, If this was the result of over fitting, then we should have lower training error in 56-layer CNN but then it also has higher training error. After analyzing more on error rate the authors were able to reach conclusion that it is caused by vanishing/exploding gradient (as the depth of CNN increases, information about the gradient passes through many layers, and it can vanish or accumulate large errors by the time it reaches the end of the network).

### Residual Block

In order to solve the problem of the vanishing/exploding gradient, this architecture introduced the concept called Residual Network. In this network we use a technique called skip connections . The skip connection skips training from a few layers and connects directly to the output.

The approach behind this network is instead of layers learn the underlying mapping, we allow network fit the residual mapping. So, instead of say H(x), initial mapping, let the network fit, F(x)= H(x)–x which gives H(x)=F(x)+x.

<p align="center">
  <img src="https://i.ibb.co/cTkwsZt/residual-block.png">
</p>

The advantage of adding this type of skip connection is because if any layer hurt the performance of architecture then it will be skipped by regularization. So, this results in training very deep neural network without the problems caused by vanishing/exploding gradient. 

### ResNet Network Architecture

The first ResNet architecture was the Resnet-34, which involved the insertion of shortcut connections in turning a plain network into its residual network counterpart. In this case, the plain network was inspired by VGG neural networks (VGG-16, VGG-19), with the convolutional networks having 3×3 filters. However, compared to VGGNets, ResNets have fewer filters and lower complexity. The 34-layer ResNet achieves a performance of 3.6 bn FLOPs, compared to 1.8bn FLOPs of smaller 18-layer ResNets.

It also followed two simple design rules –  the layers had the same number of filters for the same output feature map size, and the number of filters doubled in case the feature map size was halved in order to preserve the time complexity per layer. It consisted of 34 weighted layers.

The shortcut connections were added to this plain network. While the input and output dimensions were the same, the identity shortcuts were directly used. With an increase in the dimensions, there were two options to be considered. The first was that the shortcut would still perform identity mapping while extra zero entries would be padded for increasing dimensions. The other option was to use the projection shortcut to match dimensions. Comparison of plain network i ResNet is shown below:

<p align="center">
  <img src="https://i.ibb.co/kGFPw9M/resnet-vs-plain.png">
</p>

Various improvements of an original ResNet34 are developed through the years. One example is Resnet50 architecture which introduced one major difference. In the case of ResNet50, the building block was modified into a bottleneck design due to concerns over the time taken to train the layers. This used a stack of 3 layers instead of the earlier 2.

Therefore, each of the 2-layer blocks in Resnet34 was replaced with a 3-layer bottleneck block, forming the Resnet 50 architecture. This has much higher accuracy than the 34-layer ResNet model. The 50-layer ResNet achieves a performance of 3.8bn FLOPS!

Now we will implement ResNet34 and ResNet50 for the task of single and multiple label classification. 

Training will take a lot of time, you may go and grab some coffee while waiting... 

# Single lable classification

For this task, we will use the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) that contains images of cats and dogs of 37 different breeds. We will first show how to build a simple cat-vs-dog classifier.

More information about specific code and tasks are in corresponding Jupyter notebook. 

