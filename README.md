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
LeNet Model Architecture along with description for Traffic Signal classifier

What was the first architecture that was tried and why was it chosen?

The LeNet architecture accepts a 32x32xC image as input, where C is the number of color channels. Since traffic signal images are converted into grayscale during pre-processing of data, C is 1 in this case. But I choose to keep it 3 as i decided not to do greyscaling as part of my preprocessing.


What were some problems with the initial architecture?

When I tried this LeNet model on Internet images, it gave me 80% accuracy and making it learn from the bad data to predict can lead to even more robust traffic sign classifier.

How was the architecture adjusted and why was it adjusted?

Intially I used top_k, but it was applied to the logits instead of the softmax probabilities. I have applied tf.nn.softmax on digits before tf.nn.top_k which resulted in better accuracy in predictions.

Which parameters were tuned? How were they adjusted and why?
1. Intially i tried cv2.COLOR_RGB2GRAY to convert images to black and white. This helps in multiple ways - it reduces the amount of data to process, it allows the network to learn faster, because there is less complexity as well as it is easier to equalize the histogram in the next step.
2. Tried the histogram equalisation method using cv2.equalizeHist(image) so that the brightness values are equalized.
3. Then, the most important step is to normalize the values to go from -0.5 to +0.5 instead of going from 0 to 255. This helps keep the weights smaller and lets the network fit the curve faster.

What are some of the important design choices and why were they chosen?

I've used LeNet architecture as-is where I just modified output dimensions to predict for 43 classes instead of 10. I used batch size of 128 as it was working fine and I didn't find the need to modify it. I trained the model on different epochs and learning rates which I have explained in below point #5. I used AdamOptimizer as-is which has the benefits of moving averages of parameters (momentum) and converges quickly without hyper-parameter tuning requirements. The learning rate was tried with different values as in this order: 0.001, 0.009, 0.007, 0.005, 0.003, 0.001. As I was getting desired results at 0.001 as well, I kept it to this rate for my final model evaluation.



My Training Results...

```
EPOCH 1 of 5...
Training Accuracy = 0.959
Validation Accuracy = 0.924

EPOCH 2 of 5...
Training Accuracy = 0.986
Validation Accuracy = 0.951

EPOCH 3 of 5...
Training Accuracy = 0.990
Validation Accuracy = 0.958

EPOCH 4 of 5...
Training Accuracy = 0.994
Validation Accuracy = 0.963

EPOCH 5 of 5...
Training Accuracy = 0.996
Validation Accuracy = 0.970

Model saved
Test Accuracy = 0.948
```

### Issues Faced ###

If you face "ModuleNotFoundError: No module named 'cv2' mac" then use the following command to fix this
```
Fix: pip install opencv-python
```