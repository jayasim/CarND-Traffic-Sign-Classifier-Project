## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


**Build a Traffic Sign Recognition Project**


The Project
---
The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


### Writeup / README ###

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code. ####

You're reading it! and here is a link to my [project code](https://github.com/jayasim/CarND-Traffic-Sign-Classifier-Project-P2/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration ###

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually. ####

I loaded the dataset using pickle load() function. Then training, validation and test datasets were stored in numpy arrays. The shapes of these arrays were shown using numpy shape attribute. Number of training, validation and test examples were shown using len function of python. To see the number of output classes, np.unique() function was used:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file. ####


### Design and Test a Model Architecture ###

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. ####


#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data) ####


#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model. ####


## LeNet Model Architecture along with description for Traffic Signal classifier ##

#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate. ####

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem. ####

My Training Results...

EPOCH 1 of 5...
Training Accuracy = 0.954
Validation Accuracy = 0.928

EPOCH 2 of 5...
Training Accuracy = 0.986
Validation Accuracy = 0.950

EPOCH 3 of 5...
Training Accuracy = 0.986
Validation Accuracy = 0.955

EPOCH 4 of 5...
Training Accuracy = 0.993
Validation Accuracy = 0.959

EPOCH 5 of 5...
Training Accuracy = 0.995
Validation Accuracy = 0.963

Model saved
Test Accuracy = 0.946

##Issues Faced
If you face "ModuleNotFoundError: No module named 'cv2' mac" then use the following command to fix this
```
Fix: pip install opencv-python
```