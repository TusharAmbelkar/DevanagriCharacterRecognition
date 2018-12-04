# DevanagriCharacterRecognition

## Problem Statement:
* To create a classifier model to recognise handwritten devanagari characters and use the model to test a given image.
## Description of dataset:
* Data-type: Gray-scale image
* Classes: 46 (36 characters,10 digits)
* Dataset size: 92000(2000 examples of each class)
* Training data: 72000
* Test/Validation data: 20000
* Description:
  * Size: 32x32 pixels
  * Actual size: 28x28 pixels
  * Padding : 2 pixels in each direction
  * Background: black(0)
  * Character-color: white(255)
* Preprocessing Activity on Dataset:

  During each iteration of our training algorithm, we have applied following preprocessing attributes to avoid overfitting:
  * rotation_range = 50 ,
  * width_shift_range = 0.1
  * height_shift_range = 0.1
  * shear_range = 0.2
  * zoom_range = 0.2    
* Machine Learning algorithm used:

    The problem of image classification can be well handled by Convolutional Neural Networks.
* Convolutional Neural Networks:

  In machine learning, a convolutional neural network (CNN, or ConvNet) is a class of deep, feed-forward artificial neural networks that has successfully been applied to analyzing visual imagery.
    
  Convolutional networks were inspired by biological processes in that the connectivity pattern between neurons resembles the organization of the animal visual cortex. Individual cortical neurons respond to stimuli only in a restricted region of the visual field known as the receptive field. The receptive fields of different neurons partially overlap such that they cover the entire visual field.
  
  CNNs use relatively little pre-processing compared to other image classification algorithms. This means that the network learns the filters that in traditional algorithms were hand-engineered. This independence from prior knowledge and human effort in feature design is a major advantage.

Design:

    A CNN consists of an input and an output layer, as well as multiple hidden layers. The hidden layers of a CNN typically consist of convolutional layers, pooling layers, fully connected layers and normalization layers.
 
Convolutional

  Convolutional layers apply a convolution operation to the input, passing the result to the next layer. The convolution emulates the response of an individual neuron to visual stimuli.Each convolutional neuron processes data only for its receptive field.
Although fully connected feedforward neural networks can be used to learn features as well as classify data, it is not practical to apply this architecture to images. A very high number of neurons would be necessary, even in a shallow (opposite of deep) architecture, due to the very large input sizes associated with images, where each pixel is a relevant variable. For instance, a fully connected layer for a (small) image of size 100 x 100 has 10000 weights for each neuron in the second layer. The convolution operation brings a solution to this problem as it reduces the number of free parameters, allowing the network to be deeper with fewer parameters.For instance, regardless of image size, tiling regions of size 5 x 5, each with the same shared weights, requires only 25 learnable parameters. In this way, it resolves the vanishing or exploding gradients problem in training traditional multi-layer neural networks with many layers by using backpropagation.

Pooling:

Convolutional networks may include local or global pooling layers, which combine the outputs of neuron clusters at one layer into a single neuron in the next layer.For example, max pooling uses the maximum value from each of a cluster of neurons at the prior layer.Another example is average pooling, which uses the average value from each of a cluster of neurons at the prior layer

Fully connected:

Fully connected layers connect every neuron in one layer to every neuron in another layer. It is in principle the same as the traditional multi-layer perceptron neural network 

The structure of our convolutional neural network is similar to:



Instead of single convolutional layer before pool layer we have used two convolutional layer.
The pooling layer used is max-Pooling.
