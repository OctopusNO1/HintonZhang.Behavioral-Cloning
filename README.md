# **Behavioral Cloning/行为克隆** 

## README / Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with various filter sizes and depths between 24 and 64.

The model includes RELU layers to introduce nonlinearity, and the data  is normalized in the model using a Keras lambda layer. 

I also use crop layer to get interested area.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

I use middle, left and right cameras' images and corrected steering angles. I also flip the images to make data augmentation and balance the data distribution.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy I used for deriving a model architecture was transfer learning.

My first step was to use a convolution neural network model similar to the Nvidia End to End Self-driving Car CNN. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that original model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by adding the dropout layer but the validate loss is still high.

I run the simulator to see how well the car was driving around track one. The vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture.

| Layer         		|     Description	        					          | 
|:-----------------:|:-------------------------------------------:| 
| Input         		| 160x320x3 RGB image  						            | 
| Cropping 2D  	    | outputs 65x320x3                            |
| Normalize				  |	Lambda layer										            |
| Convolution 5x5   | 2x2 stride, ReLU activation                 |
| Convolution 5x5   | 2x2 stride, ReLU activation                 |
| Convolution 5x5   | 2x2 stride, ReLU activation                 |
| Convolution 3x3   | ReLU activation                             |
| Convolution 3x3   | ReLU activation                             |
| Flatten		        |                    			        						|
| Dropout  	        | Rate = 0.2                                  |
| Dense 100	        | 		                                        |
| Dense 50	        | 		                                        |
| Dropout  	        | Rate = 0.2                                  |
| Dense 10	        | 		                                        |
| Dense 1	          | 		                                        |

#### 3. Creation of the Training Set & Training Process

I use the Udacity provided data.

To augment the data sat, I also flipped images and angles. I think this would balance the data distribution and solve the left turn bias problem.

I then preprocessed this data by cropping the interested area and normalizing the images.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 
I used an adam optimizer so that manually training the learning rate wasn't necessary.
