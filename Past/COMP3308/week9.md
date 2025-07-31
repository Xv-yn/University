# Multilayer Neural Networks and Backpropagation

## Summary of Neural Networks

A neural network computaiton model is split into 3 main parts:
1. Architecture
    - Input, Hidden and Output Neurons
    - Connections between neurons
    - Neuron Model
    - Weight Initialization
2. Learning Algorithm
    - How weights adn connections are changed to facilitate learning
    - Mapping between input vectors and their classes 
3. Recall Technique
    - Once we have trained the model, how good is it when I give it a new 
      input?

### Neural Network Architecture

1. Neural network with more than 1 hidden layer

2. Feedforward network
    - Each neuron recieves input ONLY from neurons in the previous layer

3. Fully Connected Network
    - All neurons are connected with all neurons in the next layer

4. Weights are initialized to small random values

5. Each neuron computed weighted sum of the inputs then then applies a 
   differentiable transfer function.
   - A differentiable transfer function is simple a function that is:
     - differentiable (meaning, you can calculate its derivative)
     - and a transfer function (meaning, a function that modifies 
       the weighted sum calculated by the neuron before sending it 
       to the next layer)
        - Sigmoid squashes any input to between (0,1). 
        - Tanh squashes inputs to between (−1,1). 
        - ReLU zeroes out negative numbers and keeps positives as-is.

> [!note]
> Input neurons do not perform any computation

#### Number of Input Neurons

This depends on the input data:
- For numerical data, 1 input neuron is enough for each variable
- In the case of categorical data, num_input_neurons = num_categories
    - Number of input Neurons = Number of Categories in 1 variable
- If it helps imagine a table, it has two columns, 1 numerical and 1 
  categorical column. The numerical column will only need 1 neuron, while 
  the categorical will need numer_of_categories neurons.

#### Inputer of Output Neurons

This depends on what we want as the output. In most cases, its the types of 
outputs we want. For example, if we want a basic yes or no, then its 2 neurons.

### How many Hidden Layers do we add?

Usually trial and error.

> [!note]
> More does not always mean better!

- Too many hidden layers will lead to over fitting
- Too little hidden layers will lead to under fitting

### Backpropagation

This is basically an extension of supervised learning.

So the entire process is basically:
1. Conduct Supervised learning
2. Do Backpropagation
    - run data point 1 as input and compare expected output against 
      actual output
    - calculate the error and if accuracy is < 80% then:
        - readjust weights and re-run data point
    - After accuracy is > 80%:
        - Move onto next datapoint

Usually error is calculated with Sum of Squares Errors

In as sense, backpropagation learning can be viewed as an optimization 
search in the weight space

### Visualizing Error

When plotted into a 3D graph, we can see a sheet with multiple mountains and 
valleys. Each valley and mountain would be a local minimmum/maximum.

Now we want to move to the areas with the highest error (local mimum/maximum) 
and minimize them!

## Auto Encoders

These are basically a neural network. But with 1 hidden layer.

Basically it "encodes" an input into a smaller form then "decodes" it back 
into its "original state".

It does this by recognising patterns, its basically an AI that learns by 
recognising certain patterns and breaking down these inputs into "enncoded" 
formats, then uses these "encoded formats" to reconstruct the input as 
accurately as possible.

### Convolutions

Convolutions are an addition to the Autoencoders. Like a "plugin".

The input is sent into the autoencoder, and the "first layer" (the 
convolution) identifies the patterns. Then it is "encoded" then "decoded".

Convolutions are basically a magnifying glass looking at certain sections of 
an input. What it does, is that it uses the magnifying glass to find a 
collection of patterns that are noticed. 

These patterns aren't stored individually but are combined into a new version 
of the image that highlights important features while ignoring unimportant 
ones.

This new, pattern-rich version is passed into the encoder part of the 
autoencoder, which compresses the information even further — like 
summarizing the whole image using just a few key ideas.
Then the decoder takes that summary and tries to rebuild the original image, 
as closely as possible.

### Pooling

Now because convolution pays attention to the smaller details, we also need 
to pay attention to the bigger picture.

Pooling slightly increases the magnification range (so we can see more) and 
keep track of where these details/patterns are.

### Process

Input -> Convolution -> Pooling -> Convolution -> Pooling -> Encoding -> Decoding

> [!note]
> Normally images are NOT just black and white but colored! Hence, we set up 
> the neural network such that it takes all the red from an image as 1 input
> all the blue as another input and all the green as another input.
> This leads to a Multi-Channel Input Neural Network.
> Basically we split the image into its reds, greens and blues and use them 
> as "separate" inputs.

## Drop Out

This is basically turning off certain neurons randomly during training. So 
when training, we can't use a certain neuron in that iteration.

This is done to prevent the model from memorizing the data (overfitting).

Basically lets say we want to determine if a number is divisible by 3. If 
we only focus on the right-most digit, e.g. XXX3, we can determine if that 
number is divisible by 3. However, that's not nesessarily true, e.g. 23. 

So, by "turning" off "that" neuron, we are forced to look at the other digits 
as well to determine if the number is divisible by 3.

