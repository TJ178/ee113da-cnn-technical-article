# On the Subject of Convolutional Neural Networks
Timothy Jacques, UID 005537618

## Introduction

As the sheer quantity of computational resources easily available increases, we find more and more computationally heavy ways to model the world around us. For most phenomena, modeling through traditional methods is very simple. Most basic systems can be modeled through just mathematical equations, where generating a predicted output is as simple as changing the inputs and reevaluating the equations. Performing these operations using a computer would be as straightforward as implementing the same mathematical function in code. 

However, there are many systems where mathematically calculating a result is not feasible. These systems may have models that are too inaccurate, too complex, or they may not have a known mathematical model at all!

One of the tools that we can use to model these systems is something called a **neural network**.

## Neural Networks
Neural networks are a type of **machine learning** model that is founded upon the organization of neurons in organisms. That is, these networks follow a similar organization to the neurons in your brain, giving them capable learning abilities.

### Structure

![image info](https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/250px-Colored_neural_network.svg.png)

**Figure 1:** The basic structure of a neural network, displaying connected layers of nodes called neurons.<sup>[[3]](#references)</sup>

As seen in the image above, a neural network is structured in layers that consist of nodes called **neurons** and connecting **edges**. This structure contains an input layer, some number of hidden layers, and then an output layer.

Starting from the input layer, the input neurons are populated with values that they send out to each of the connected edges. These edges each have a distinct **weight**, where the value that crosses them is scaled by a certain amount.

In the hidden layers, each neuron takes in inputs from the connected edges, performs some **activation function** on those inputs, then sends the output to the next layer. These activation functions typically are non-linear, allowing them to approximate any function, given the right inputs.<sup>[[1]](#references)</sup> Additionally, each hidden layer can perform a different operation, allowing layers to be chained together and emulate a more complex system.

Once the values have passed through all of the hidden layers, another set of weighted edges carry the values to the final output layer.

### Training

Now that we know how neural networks are structured, how do we **train** a network to solve a specific problem? When training a network, a dataset with known inputs and outputs is fed into the model. The output of the model for a certain input is then compared with the true output to get an error. This error is then used to adjust the **weights** of the connecting edges described in the previous section and decrease the error. This operation continues until the error ceases to decrease, at which point the model is trained as well as its architecture allows.<sup>[[2]](#references)</sup>

### Model Tuning
When designing a neural network for a specific task, the complexity of the problem changes the number of resources dramatically. For example, the amount of input information could be extremely large, requiring the total number of neurons to be much larger. In addition, more complex problems could require more hidden layers to accurately classify, leading to more necessary weights. 

When designing a network, these parameters that are changed are called **hyperparameters**. With more complex models, there may be dozens of different hyperparameters to change, and finding the most optimal solution becomes the main obstacle for an efficient and accurate model. 

However, with large inputs, using our existing neural network model will become unwieldly quickly, even with an optimized model. As the input size increases, the number of edges and nodes will increase dramatically. 

Therefore, for applications that process a lot of data, such as image processing, a typical neural network is too heavy to be practical. We will now discuss a solution to this issue.

## Convolutional Neural Networks
A convolutional neural network is a specific type of neural network that is widely used for image and video processing due to its ability to readily handle the large amounts of data associated with color images.

A convolutional neural network (hereon referred to as a CNN), is mainly defined by the three different types of hidden layers that it consists of: **convolutional** layers, **pooling** layers, and **fully-connected** layers. These will be discussed in detail shortly.

The main feature that allows CNNs to work well with large input data is the fact that the process of **convolution** compresses the information that needs to be processed, decreasing the number of weights and the overall size of the network drastically.<sup>[[2]](#references)</sup>

Additionally, CNNs are typically fast, due to their use of a rectified linear unit activation function.

### Rectified Linear Unit (ReLu)
CNNs typically use the **rectified linear unit** activation function, or "ReLu" function. The function is as follows:
$$f(x)=\text{max}(0, x)$$
This function effectively removes negative outputs while allowing positive outputs to be as high as possible. ReLu is preferred due its simplicity, which allows for much faster calculation while retaining accuracy.<sup>[[4]](#references)</sup> 

This function is used heavily throughout the layers of a CNN, which we will now discuss.

### Convolutional Layer
The first, and most notable layer is the convolutional layer. In addition to the learned edge weights, the convolutional layer also utilizes a **kernel** (or filter) that is used to both perform an operation on the input while also *significantly decreasing* the amount of output data. 

The kernel is a relatively small filter that is *convolved* across the entirety of the input. That is, it moves across the input data, multiplies itself elementwise with the filtered input data, then calculates the sum of the products to obtain a single value. This value is next passed through the activation function (usually [ReLu](#rectified-linear-unit-relu)) and then to the next layer. 

This operation is visualized below in Figure 2.

![image](https://www.ibm.com/content/dam/connectedassets-adobe-cms/worldwide-content/creative-assets/s-migr/ul/g/ed/92/iclh-diagram-convolutional-neural-networks.png)

**Figure 2**: Illustration of convolution, showing the input image, the filter, and the output contents.<sup>[[8]](#references)</sup>

As a side note, for inputs that are multidimensional, the kernel usually spans the entire depth of the input, and only convolves across two dimensions.

Through this movement, the convolutional layer can decrease the amount of data that is passed to the next layer significantly, enabling the model to take in much more data than a traditional neural network can.

As an example of the efficiency of CNNs, we can compare the total number of weights that are necessary for a certain input size, say 128x128, for a 128x128 monochrome image. With a traditional fully-connected neural network, this would require 1 weight per input value, or 16384 weights per neuron. A comparable CNN with a kernel size of 4x4 would reduce this to 16 weights per neuron.<sup>[[2]](#references)</sup>

#### Convolutional Layer Hyperparameters
When designing a convolutional neural network, there are a few different hyperparameters that can be modified. 
- **Depth**: For multidimensional inputs, the depth of the kernel defines the amount that the depth of the output is decreased.<sup>[[2]](#references)</sup>
- **Stride**: This defines the amount that the kernel moves during each iteration while convolving across the input. With lower stride, the amount of overlap is large, and the output value is larger.<sup>[[2]](#references)</sup>
- **Zero-Padding**: This defines the amount of 0-valued border that is applied to the input. This allows for greater control over the output size of the convolutional layer.<sup>[[2]](#references)</sup>

Each of these hyperparameters provide a tradeoff between model accuracy and model complexity.

### Pooling Layer
While convolution can greatly reduce the number of neurons, another layer is typically used to decrease the data even more.

Pooling layers very simply perform downsampling of the input through various methods. One of the most common types of pooling is **max pooling**, where groups of a certain size are reduced to one datapoint by taking the max of the entire group. Other forms of pooling include min and average pooling, which perform similar downsampling operations. 

Pooling layers are necessary to decrease the amount of parameters in the next layer, decreasing the size of the whole model while retaining the most important data from each group.<sup>[[2]](#references)</sup>

The pooling layer only has the pooling size hyperparameter, which determines the number of elements that are merged into one. Typically, decreasing the pooling size leads to more resource usage, but potentially higher accuracy.

### Fully-Connected Layer
After processing the input data down to a much more concentrated state, it still needs to be classified. Now that the information is concentrated, we can simply use this information as the input to a typical neural network that is fully-connected, which is a much more reasonable size due to the smaller input.

These layers are most similar to regular neural networks, as described in the [previous sections](#neural-networks), and contain neurons that are fully connected to the previous and next layers. In CNNs, the activation function used is also typically [ReLu](#rectified-linear-unit-relu).<sup>[[2]](#references)</sup>

### Convolutional Neural Network Structure
Now that we know each of the layers, we will discuss the structure of a typical CNN. 

To constitute a CNN, there is usually at least one of each type of layer. For more complex problems, the structure usually includes two convolutional layers followed by a pooling layer, repeated until one or many fully-connected layers.<sup>[[2]](#references)</sup>

![image info](https://upload.wikimedia.org/wikipedia/commons/6/63/Typical_cnn.png)

**Figure 3**: Structure of a CNN with two subsequences of a convolutional layer and pooling layer, then lastly connected with a fully-connected final layer.<sup>[[7]](#references)</sup>

## Conclusion
With the information from above, we have learned about the structures of neural networks and the basics upon how they work. We learned that for applications such as image processing, creating a typical, fully-connected neural network is unreasonable due to the sheer quantity of data. With this in mind, we learned that convolutional neural networks help alleviate this problem by compressing usable information into a more reasonable input for a regular neural network.

Because of the advent of CNNs, image classification can be much lighter than previously, allowing it to be run on devices as small as a Raspberry Pi. With this availability, the applications of image classification are much more in reach.

## References
 [1] Cybenko, G. (December 1989). "Approximation by superpositions of a sigmoidal function" (PDF). Mathematics of Control, Signals, and Systems. 2 (4): 303–314. doi:10.1007/BF02551274. ISSN 0932-4194. S2CID 3958369.

 [2] O'Shea, Keiron, and Ryan Nash. "An introduction to convolutional neural networks." arXiv preprint arXiv:1511.08458 (2015).

 [3] Glosser.ca. “File:Colored Neural Network.Svg.” Wikimedia Commons, 28 Jan. 2013, commons.wikimedia.org/wiki/File:Colored_neural_network.svg. 

 [4] Krizhevsky, A.; Sutskever, I.; Hinton, G. E. (2012). "Imagenet classification with deep convolutional neural networks" (PDF). Advances in Neural Information Processing Systems. 1: 1097–1105. Archived (PDF) from the original on 2022-03-31. Retrieved 2022-03-31.

 [5] Ciresan, D., Meier, U., Schmidhuber, J.: Multi-column deep neural networks for image classification. In: Computer Vision and Pattern Recognition (CVPR), 2012 IEEE Conference on. pp. 3642–3649. IEEE (2012)

[6] Ciresan, D.C., Meier, U., Masci, J., Maria Gambardella, L., Schmidhuber, J.: Flexible, high performance convolutional neural networks for image classification. In: IJCAI Proceedings-International Joint Conference on Artificial Intelligence. vol. 22, p. 1237 (2011)

[7] By Aphex34 - Own work, CC BY-SA 4.0, https://commons.wikimedia.org/w/index.php?curid=45679374

[8] “What Are Convolutional Neural Networks?  | IBM.” IBM, www.ibm.com/topics/convolutional-neural-networks. Accessed 16 Nov. 2023.