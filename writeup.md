#**Traffic Sign Recognition** 

##Writeup 

---


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set 
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./data_set_distribution.png "Visualization"
[image2]: ./new_image/11.png "Traffic Sign 1"
[image3]: ./new_image/12.png "Traffic Sign 2"
[image4]: ./new_image/13.png "Traffic Sign 3"
[image5]: ./new_image/18.png "Traffic Sign 4"
[image6]: ./new_image/34.png "Traffic Sign 5"
[image7]: ./new_image/36.png "Traffic Sign 6"
[image8]: ./new_image/38.png "Traffic Sign 7"
[image9]: ./new_image/39.png "Traffic Sign 8"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/liruixuan-xidian/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier_new.ipynb)

###Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 4410
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. It is a bar chart showing how the datadistribute. 

![alt text][image1]

###Design and Test a Model Architecture

As a first step, I decided to convert the images to grayscale because the experience shows that the gray level have much influence in image recognition. The gray image also have less dimension which could have less compute complexity.

As a second step, I normalized the image data so that I could speed up the gradient descent algorithm.


My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Dropout               | Keep_prob 0.5                                 |
| Max pooling	      	| 2x2 stride,  outputs 16x16x6  				|
| Convolution 3x3	    | 1x1 stride,  outputs 10x10x16      		    |
| RELU                  |                                               |
| Dropout               | Keep_prob 0.5                                 |
| Max pooling           | 2x2 stride,  outputs 5x5x16                   |
| Flatten               | Output 400                                    |
| Fully connected		| Output 120  									|
| RELU                  |                                               |
| Fully connected       | Output 84                                     |
| RELU                  |                                               |
| Fully connected       | Output 43                                     |
 

To train the model, I used an optimizer called tf.train.AdamOptimizer. The batch size has been set to 64 and number of epochs has been set to 50. Learning rate is 0.001.

My final model results were:
* training set accuracy of 99.8%
* validation set accuracy of 93.9% 
* test set accuracy of 93.1%

according to the Udacity advice, I grayscalize and nomarlize the original data, when I use formulate x-128/128. the accurate is much lower than 80%, so I change the formulate to x/255-0.5. Then the accuracy is improved but still less than 90%. I try to change the value of batchsize,learning rate,epoch, but it is useless. so I realize the model need to be modified. and I add a dropout layer between convolution layer. at last I get accuracy of 93.9% in validation set and accuracy of 93.1% in test set.


###Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4] 
![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9]

The mistake this modle had made in new image set is Keep left sign which had been predict as Speed limit. This image has a black background which could be difficult to distinguish in gray level. and also a lot of noisiness included in the middle of this picture, all of this make this image difficult to classify.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Turn left ahead  		| Turn left ahead								| 
| General caution		| General caution						    	|
| Keep right			| Keep right									|
| Yield	      		    | Yield					 				        |
| Right-of-way          | Right-of-way                                  |
| Keep left             | Speed limit(70km/h)                           |
| Priority road         | Priority road                                 |
| Go straight or right  | Go straight or right                          |

The model was able to correctly guess 8 of the 9 traffic signs, which gives an accuracy of 87.5%. 

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

For the sixth image, the model predict the data as Speed limit(probability of 45%), but the image is Keep left. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .45         			| Speed limit(70km/h)   		        		| 
| .40     				| Keep left										|
| .10					| Speed limit(50km/h)							|
| .01	      			| Speed limit(120km/h)                          |
| .008				    | Speed limit(30km/h)                           |


For the eight image, the model predict the data as Go straight or right(probability of 72%).The top five softmax probabilities were
  
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .72         			| Go straight or right   		        		| 
| .11     				| Traffic signals                               |
| .09					| General caution   							|
| .03	      			| No vehicles                                   |
| .02				    | Turn left ahead                               |

For the other images, every images has been predict correctly and probability of more than 99%.


