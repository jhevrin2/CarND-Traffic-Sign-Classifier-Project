# **Traffic Sign Recognition** 

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

#### Writeup / README

1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Link to my [project code](https://github.com/jhevrin2/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).

#### Data Set Summary & Exploration

1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32
* The number of unique classes/labels in the data set is 43

2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. My main goal was to see the distribution of the target.  Here is a histogram of the signs in the data.

![distribution](https://github.com/jhevrin2/CarND-Traffic-Sign-Classifier-Project/blob/master/figures/distribution.png)

Due to this, I will augment the lesser represented signs doing a random rotation.  In addition, I did poke around in the dataset (shown in the notebook) to see what some of the images looked like (given my knowledge of German traffic signs is limited).

#### Design and Test a Model Architecture

1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

* As stated above, I wanted to augment signs with less representation.  I simply did a random (very small) rotation of the image using OpenCV.  

Before:  ![before](https://github.com/jhevrin2/CarND-Traffic-Sign-Classifier-Project/blob/master/figures/crossing.png)
After:  ![after](https://github.com/jhevrin2/CarND-Traffic-Sign-Classifier-Project/blob/master/figures/crossing-rotate.png)

* Here is the new distribution with more images (no sign has less than 1000):

![distribution_new](https://github.com/jhevrin2/CarND-Traffic-Sign-Classifier-Project/blob/master/figures/distribution_new.png)

* After this, I chose to grayscale.  This can help to remove noise and make it easier for the model to learn the different markings of the sign.

Before:  ![before](https://github.com/jhevrin2/CarND-Traffic-Sign-Classifier-Project/blob/master/figures/crossing.png)
After:  ![after](https://github.com/jhevrin2/CarND-Traffic-Sign-Classifier-Project/blob/master/figures/crossing-gray.png)

* Finally, I normalized the image.  This will help to center and scale the image to assist with gradient decent.

2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I mostly used the LeNet archticture from the lab assignment.  I did add a couple of max pooling transformations to hopefully bring out some of the features in the image.  I had looked at adding in some dropout, however, my training accuracy was never high enough to think that I was overfitting.  Here's my final set up:

* Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
* Activation
* Pooling. Input = 28x28x6. Output = 14x14x6. Stride = 2x2
* Layer 2: Convolutional. Output = 10x10x16.    
* Activation.
* Pooling. Input = 10x10x16. Output = 5x5x16. Stride = 2x2
* Flatten. Input = 5x5x16. Output = 400.
* Layer 3: Layer 3: Fully Connected. Input = 400. Output = 120.    
* Activation.
* Layer 4: Fully Connected. Input = 120. Output = 100.  
* Activation.
* Layer 5: Fully Connected. Input = 100. Output = 43.

3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I ran with the following configuration (mostly from LeNet example):
* Epochs = 25
* Batch size = 128
* Learning rate = 0.001
* Optimizer = AdamOptimizer

4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I took MANY different steps to get to this 

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.951
* test set accuracy of 0.923

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

I chose to use the LeNet architecture as my base.  I didn't deviate heavily from this given it was giving me solid performance.

* What were some problems with the initial architecture?

At first it wasn't fitting the training data well given the class imbalance.  After this, it started to fit the data much better.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

I did add some max pooling to call out the features in the image better with the hopes of 

* Which parameters were tuned? How were they adjusted and why?

I added additional epochs to train the model longer and learn the training set better.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Again, I chose to base my architecture off the LeNet architecture.  This leverage CNNs (good for images) and works well as a starting point for image classification problems.  I could've added dropout to help with generalization and avoiding overfitting on the training set.

If a well known architecture was chosen:
* What architecture was chosen?  

Based off of LeNet, but added some max pooling layers.

* Why did you believe it would be relevant to the traffic sign application?

See above.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
It's fitting the training set well, but slightly overfitting given the test and validation accuracy is a bit lower.  I could've added some dropout or early stopping to avoid such overfitting, but given the task, the model meets the goal.

#### Test a Model on New Images

1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

<img src="https://github.com/jhevrin2/CarND-Traffic-Sign-Classifier-Project/blob/master/tests/bump.jpg" width="250px" height="250px">
<img src="https://github.com/jhevrin2/CarND-Traffic-Sign-Classifier-Project/blob/master/tests/do_not_enter.jpg" width="250px" height="250px">
<img src="https://github.com/jhevrin2/CarND-Traffic-Sign-Classifier-Project/blob/master/tests/kph_30.jpg" width="250px" height="250px">
<img src="https://github.com/jhevrin2/CarND-Traffic-Sign-Classifier-Project/blob/master/tests/stop.jpg" width="250px" height="250px">
<img src="https://github.com/jhevrin2/CarND-Traffic-Sign-Classifier-Project/blob/master/tests/yield.jpg" width="250px" height="250px">

2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bump      		| Left turn ahead   									| 
| Do Not Enter     			| Do Not Enter 										|
| 30 KPH					| 30 KPH											|
| 100 km/h	      		| Bumpy Road					 				|
| Yield			| Yield      							|

The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This is a bit lower than the test set accuracy, but given the low sample size, this could happen.

3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

.99788082e-01, .0002, 3.8998827e-05, 3.1032174e-05,
        1.5908500e-07

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Left turn ahead   									| 
| .0002     				| Bumpy Road 										|
| .000004					| Bicycle Crossing											|
| .000003	      			| Curve Right					 				|
| .00000007				    | Ice/Snow      							|

Please see the end of the [notebook](https://github.com/jhevrin2/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb). for additional details on other images
