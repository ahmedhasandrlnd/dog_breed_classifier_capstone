 

## Introduction
 

# Dog Breed Classifier

<a id='index'></a>
## Table of Contents
- [Project Overview](#overview)
- [Problem Statement](#statement)
- [Metrics](#metrics)
- [Data Exploration and Visualization](#explore)
- [Data Preprocessing](#preprocess)
- [Implementation](#implement)
  - [Step 1](#step1): Detect Humans
  - [Step 2](#step2): Detect Dogs
  - [Step 3](#step3): Create a CNN to Classify Dog Breeds (from Scratch)
  - [Step 4](#step4): Use a CNN to Classify Dog Breeds (using Transfer Learning)
  - [Step 5](#step5): Create a CNN to Classify Dog Breeds (using Transfer Learning)
  - [Step 6](#step6): Write your Algorithm
- [Refinement](#refine)
- [Model Evaluation and Validation](#eval)
- [Justification](#just)


  - [Step 7](#step7): Test Your Algorithm
  - [Step 8](#flask): Flask Web App
  	- [Dog Detector](#dog)
  	- [Human Detector](#human)
  	- [Unknown Detector](#unknown)
  	- [TopK Prediction](#topk)
- [Usage Instructions](#run)
  - [Data Cleaning](#cleaning)
  - [Training Classifier](#training)
  - [Starting the Web App](#starting)
  	- [Running in localhost](#local)
  	- [ngrok Deployment](#ngrok)
- [Conclusion](#conclusion)
	- [Reflection](#reflection)
	- [Improvement](#improvement)
- [Software Requirements](#sw)
- [Files](#files)
- [Credits and Acknowledgements](#credits)


<a id='video'></a>
**_Video Demo of the deployed App_**
![Disaster Response App](gif/disaster.gif)
<hr/> 

[Back to Table of Content](#index)


<a id='overview'></a>
## Project Overview
<img src="images/Brittany.jpg" width="100"> | <img src="images/Welsh_springer_spaniel.jpg" width="200">
<p> If you look at the two dog pictures above, you might thought they are the same breed. But, you would be surprised to know that they are quite different. The left one is Brittany, and the right one Welsh Springer Spaniel. There are thousands of different dog breeds in the world. Some of them are so visually distinct  that it is easily possible to tell their breeds from their images. But, for some breeds like the above image pair, it is quite difficult to distinguish them. In order to solve this problem, I want to leverage the state of art of image classification model on Imagenet to teach computer to predict dog's breed from an image by training it using 6680 images of 133 breeds, 835 images for validation and 836 images for testing. <br/>

[Back to Table of Content](#index)

<a id='statement'></a>
## Problem Statement
 Aside from the big problem of dog breed classification, I would like to tackle two minor but interesting problems as well such as human face detection and dog detection using computer vision and machine learning techniques. Using these three solutions, I would like to build an application that could be used as part of a mobile or web app. This application  will accept any user-supplied image as input. If a dog is detected in the image, it will provide an estimate of the dog's breed. If a human is detected, it will provide an estimate of the dog breed that is most resembling. If it detects neither a dog nor a human, it will give an appropriate error message. <br/>
<hr/> 

[Back to Table of Content](#index)

<a id='metircs'></a>
## Metrics
Since we are dealing with a multi-classification problem here and the data is slightly imbalanced, I used accuracy evaluation metric and negative log-likelihood loss function. The main objective in a learning model is to reduce (minimize) the loss function's value with respect to the model's parameters by changing the weight vector values through different optimization methods, such as backpropagation in neural networks.

Loss value implies how well or poorly a certain model behaves after each iteration of optimization. Ideally, one would expect the reduction of loss after each, or several, iteration(s).

The accuracy of a model is usually determined after the model parameters are learned and fixed and no learning is taking place. Then the test samples are fed to the model and the number of mistakes (zero-one loss) the model makes are recorded, after comparison to the true targets. Then the percentage of misclassification is calculated.

For example, if the number of test samples is 1000 and model classifies 950 of those correctly, then the model's accuracy is 95.0%. [Ref](https://stackoverflow.com/questions/34518656/how-to-interpret-loss-and-accuracy-for-a-machine-learning-model)

Also, we have employed 3 types of error metrics namely top-1, top-3 and top-5 error. In the case of top-1 error, we check if the top class (the one having the highest probability) is not the same as the target label.

In the case of top-3 errore, we check if the target label is not one of our top 3 predictions (the 3 ones with the highest probabilities)

In the case of top-5 error, we check if the target label is not one of our top 5 predictions (the 5 ones with the highest probabilities). 
<br/>
<hr/> 

[Back to Table of Content](#index)

<a id='explore'></a>
## Data Exploration and Visualization
From the train, test and valid dataset of dog images, we get the following distribution and statistics:
![train_dist](images/train_dist.PNG)
![test_dist](images/test_dist.PNG)
![valid_dist](images/valid_dist.PNG)

![stat](images/stat.PNG)

<hr/> 

[Back to Table of Content](#index)

<a id='preprocess'></a>
## Data Preprocessing
Each image in the train, valid and text dataset goes thorugh a number of preprocessing steps:

1. Train and Valid Dataset images goes through a series of transforms.RandomRotation(10), transforms.Resize(256), transforms.CenterCrop(224),transforms.RandomHorizontalFlip()
1. Test Dataset images goes through a series of transforms.Resize(256) and transforms.CenterCrop(224) 
1. Normalize: Each image is normalized by mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]

<hr/> 

[Back to Table of Content](#index)


<a id='implement'></a>
## Implementation
The implementation of the project is divided into following steps:

<a id='step1'></a>
### Step 1: Detect Humans

We use OpenCV's implementation of Haar feature-based cascade classifiers to detect human faces in images. OpenCV provides many pre-trained face detectors, stored as XML files on github. We have downloaded one of these detectors and stored it in the haarcascades directory.<br/>
Before using any of the face detectors, it is standard procedure to convert the images to grayscale. The detectMultiScale function executes the classifier stored in face_cascade and takes the grayscale image as a parameter. 

<hr/> 

[Back to Table of Content](#index)

<a id='step2'></a>
### Step 2: Detect Dogs
We use a pre-trained VGG16 model to detect dogs in images. Our first line of code downloads the ResNet-50 model, along with weights that have been trained on ImageNet, a very large, very popular dataset used for image classification and other vision tasks. ImageNet contains over 10 million URLs, each linking to an image containing an object from one of 1000 categories. Given an image, this pre-trained VGG16 model returns a prediction (derived from the available categories in ImageNet) for the object that is contained in the image.<br/>
While looking at the [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a), you will notice that the categories corresponding to dogs appear in an uninterrupted sequence and correspond to dictionary keys 151-268, inclusive, to include all categories from `'Chihuahua'` to `'Mexican hairless'`.  Thus, in order to check to see if an image is predicted to contain a dog by the pre-trained VGG16 model, we need only check if the `ResNet50_predict_labels` function above returns a value between 151 and 268 (inclusive).

[Back to Table of Content](#index)

<a id='step3'></a>
### Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
Now that we have functions for detecting humans and dogs in images, we need a way to predict breed from images. In this step, we create a CNN that classifies dog breeds. We create our CNN from scratch to attain a test accuracy of at least 1%.

[Back to Table of Content](#index)

<a id='step4'></a>
### Step 4: Use a CNN to Classify Dog Breeds (using Transfer Learning)
To reduce training time without sacrifysing accuracy, we have used the pre-trained VGG-16 model as a fixed feature extractor, where the last convolutional output of VGG-16 is fed as input to our model. We only add a global average pooling layer and a fully connected layer, where the latter contains one node for each dog category and is equipped with a softmax.

[Back to Table of Content](#index)

<a id='step5'></a>
### Step 5: Create a CNN to Classify Dog Breeds (using a different Transfer Learning)- Refinement
In Step 5, we used transfer learning to create a CNN using VGG-19 bottleneck features. In this section, we used the bottleneck features from a different pre-trained model such as VGG19. 

[Back to Table of Content](#index)


<a id='step6'></a>
### Step 6: Write your Algorithm
In this step, we write an algorithm that accepts a file path to an image and first determines whether the image contains a human, dog, or neither. Then,
* if a dog is detected in the image, return the predicted breed.
* if a human is detected in the image, return the resembling dog breed.
* if neither is detected in the image, provide output that indicates an error.

[Back to Table of Content](#index)

<hr/> 

[Back to Table of Content](#index)


<a id='refine'></a>
## Refinement
In our first design, we have used transfer learning to create a CNN using VGG-16 bottleneck features. As a refinement, in our second model we have used transfer learning to create a CNN using VGG-19 bottleneck features. Second model gives slightly better results with better performance metrics. Please refer to the [next section](#eval) for a detailed comparison. Here are some missclassified images (top-5 error images) from the first model prediction:
![top-5](images/model1_top5.PNG)

Here are some missclassified images (top-5 error images) from the second model prediction:
![top-5](images/model2_top5.PNG)

From these two figures, we can see that out of these 9 images, 5 images were misclassified by both models. But each model made some unique mistakes as well from the other model.

<br/>
<hr/> 

[Back to Table of Content](#index)


<a id='eval'></a>
## Model Evaluation and Validation
Now, we are going to show the comparison of performance metrics of two of our models using the transfer learning techniques. The first model used VGG-16 bottleneck features and the second model used VGG-19 bottleneck features. Out of 836 test images, the first model misclassified 128 images as top-1 error, 21 images as top-3 error and 10 images as top-1 error which gives a top-1 error rate of 15.3%, top-3 error rate of 2.5% and top-5 error rate of 1.19%. On the contrary, the second model misclassified 110 images as top-1 error, 17 images as top-3 error and 9 images as top-1 error which gives a top-1 error rate of 13.5%, top-3 error rate of 2.05% and top-5 error rate of 1.07%.

|         Model          |      Accuracy     | Top-1 Error Rate | Top-3 Error Rate | Top-5 Error Rate |
| :--------------------: | :---------------: | :--------------: | :--------------: | :--------------: | 
|    Pretrained VGG16    |      84.70%       |      15.30%      |      2.50%       |      1.19%       |
|    Pretrained VGG19    |      86.85%       |      13.15%      |      2.03%       |      1.07%       |

The performance can be further improved by using some other pretrained model such as Densenet, Resnet or Inception models.
<br/>

<a id='just'></a>
## Justification
Due to less number of dog images of certain breeds, the model finds it difficult to predict some breeds.  
We have observed that the model couldn’t classify between Great pyrenees and Kuvasz, which both are white, big, and fluffy. <br/>
[Great pyrenees]<img src="images/Great_pyrenees.jpg" width="20%">| [Kuvasz]<img src="images/Kuvasz.jpg" width="30%">
Also we have found the model fails to correctly classify german wirehaired pointer and wirehaired pointing griffon which look quite similar. <br/>
[german wirehaired pointer]<img src="images/German_wirehaired_pointer.jpg" width="20%">| [wirehaired pointing griffon]<img src="images/Wirehaired_pointing_griffon.jpg" width="30%">
It also couldn’t distinguish between Mastiff and Bullmastiff, which is a mix between a bulldog and a mastiff.
<br/>
[Bulldog]<img src="images/Bulldog.jpg" width="20%">|[Mastiff]<img src="images/Mastiff.jpg" width="20%">|[Bullmastiff]<img src="images/Bullmastiff.jpg" width="20%"> 
<hr/> 

[Back to Table of Content](#index)







<a id='step7'></a>
### Step 7: Test Your Algorithm
In this step, we test our algorithm at least six images on our computer. We've used at least two human and two dog images. 

[Back to Table of Content](#index)

<a id='flask'></a>
### 2.3. Flask Web App
The Flask web app serves two purposes:

- Dataset Visualization
- Message categorization

<a id='visual'></a>
#### 2.3.1. Data Visualization
The web app provides the following visualizations related to dataset:

**_Screenshot: Distribution of message genres_**

![Distribution of message genres](gif/graph1.PNG)
<a id='imbalance'></a>
**_Screenshot: Distribution of message categories_**

![Distribution of message categories](gif/graph2.PNG)

**_Screenshot: Distribution of messages lengths_**

![Distribution of messages lengths](gif/graph3.PNG)

**_Screenshot: Correlation of message categories_**

![Correlation of message categorie](gif/graph4.PNG)

**_Screenshot: Messages and categories counts_**

![Messages and categories counts](gif/graph5.PNG)
<hr/> 

[Back to Table of Content](#index)

<a id='category'></a>
#### 2.3.2. Message  Categoriztion
The web app lets an user enter an emergency message during a natural disaster, e.g. _"We're asking for water, medical supply, food"_.

**_Screenshot: New Message_**

![New Message](gif/new.PNG)

After submitting the message, the web app will classify the message into categories so that appropriate relief agency can be reached out for help.

**_Screenshot: Message Categorization_**

![results](gif/result.PNG)
<hr/> 

[Back to Table of Content](#index)

<a id='run'></a>

## 3. Usage Instructions

There are three steps to get up and runnning with the web app if you want to start from ETL process.

<a id='cleaning'></a>

### 3.1. Data Cleaning

Go to the project directory and the run the following command:

```
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

The first two arguments are input data and the third argument is the SQLite Database in which we want to save the cleaned data. The ETL pipeline is in _process_data.py_.

**_Screenshot: Data Claening_**

![results](gif/cleaning.PNG)
<hr/> 

[Back to Table of Content](#index)

<a id='training'></a>

### 3.2. Training Classifier

After the data cleaning process, run this command from the project directory:

```
python models/train_classifier.py data/DisasterResponse.db models/model.pkl
```

This will use cleaned data to train the model, improve the model with grid search and saved the model to a pickle file (_model.pkl_).
<hr/> 

[Back to Table of Content](#index)

<a id='starting'></a>

### 3.3. Starting the web app

<a id='local'></a>
#### 3.3.1. Running in localhost

Now that we have cleaned the data and trained our model. Now it's time to see the prediction in a user friendly way.

Go the app directory and run the following command:

<a id='com'></a>

```
python disaster_prediction.py
```

This will start the web app and will direct you to a URL (_http://localhost:5000_) where you can enter messages and get classification results for it.

<a id='ngrok'></a>
#### 3.3.2. ngrok Deployment

To access our local running web app from a public endpoint such as Internet, we have used _ngrok_ which is a multiplatform tunnelling, reverse proxy software. After running the _ngrok_ program, we enter the command:

```
ngrok http 5000
``` 

This will create a temporary URL _(http://37ca28eb.ngrok.io)_ via which we can access our web app running in our local computer from other computers. See [video demo](#video).

**_Screenshot: ngrok reverse tunnelling_**

![results](gif/ngrok1.PNG)
<hr/> 

[Back to Table of Content](#index)

<a id='conclusion'></a>

## 4. Conclusion

As we can see from the [distribution of messages categories](#imbalance), the dataset is highly imbalanced. It gives a very interesting evaluation results as given below:

**_Screenshot: Evaluation Matrix_**

![results](gif/eval.PNG)

 
<hr/> 

[Back to Table of Content](#index)

<a id='reflection'></a>
### Reflection

These are the intersting or difficult aspects of this present application:
1. __GPU TRAINING__: The traing requires a lot of computational power and hence it is impossible to do the project without GPU. I have done the traing in my local laptop equipped with GTX 1060 GPU. Even though it has 6GB RAM, but it happened that when I tried to train two transfer models on a same notebook, the GPU ran out of memory. So we train different transfer models in different notebooks to overcome the situation.
1. __WEB DEPLOYMENT__: Due to some credit problem, we haven't tried to deploy our application in an external server. But we have used ngrok, which provide us a public URL for our application running in local web server. 


[Back to Table of Content](#index)


<a id='improvement'></a>
### Improvement

These are the improvement ideas on this present application:
1. __AUGMENT THE TRAINING DATA:__ Augmenting the training and/or validation set might help improve model performance.
1. __OVERLAY DOG EARS ON DETECTED HUMAN HEADS:__ Overlay a Snapchat-like filter with dog ears on detected human heads. You can determine where to place the ears through the use of the OpenCV face detector, which returns a bounding box for the face. If you would also like to overlay a dog nose filter, some nice tutorials for facial keypoints detection exist [here](https://www.kaggle.com/c/facial-keypoints-detection/details/deep-learning-tutorial).
1. __ADD FUNCTIONALITY FOR DOG MUTTS:__ Currently, if a dog appears 51% German Shephard and 49% poodle, only the German Shephard breed is returned. The algorithm is currently guaranteed to fail for every mixed breed dog. Of course, if a dog is predicted as 99.5% Labrador, it is still worthwhile to round this to 100% and return a single breed; so, you will have to find a nice balance.
1. __AWS, GCP or AZURE DEPLOYMENT__: It would be a good learning experience if we could depoly our web application to AWS, GCP or AZURE platform.


[Back to Table of Content](#index)

<a id='sw'></a>

## 5. Software Requirements
This project uses **Python 3.6.6** and the necessary libraries are mentioned in _requirements.txt_.
<hr/> 

[Back to Table of Content](#index)

<a id='files'></a>

## 6. Files
<pre>
+
|   README.md 					# Readme file
|   requirements.txt			# Required packages
|   
+---app
|   |   disaster_response.py	# Flask file that runs the app
|   |   forms.py				# Forms used in the app
|   |   site.db					# database to save user's messages
|   |   
|   +---static
|   |   |   main.css    		# main stylesheet
|   |   |   
|   |   \---msg_pics
|   |           msg_pics.zip	# image folder used in the app
|   |           
|   \---templates
|           about.html			# about page
|           create_msg.html		# new message page
|           graph1.html			# graph1 page
|           graph2.html			# graph2 page
|           graph3.html			# graph3 page
|           graph4.html			# graph4 page
|           graph5.html			# graph5 page
|           home.html			# home page
|           layout.html			# layout page
|           msg.html			# message page
|           
+---data
|       DisasterResponse.db  	# cleaned database
|       disaster_categories.csv	# given categories dataset
|       disaster_messages.csv	# given messages dataset
|       process_data.py			# performs ELT pipeline
|       
+---gif
|       Images folder used in Readme.md
|       
\---models
        train_classifier.py		# performs classification task
</pre>
<hr/> 

[Back to Table of Content](#index)

<a id='credits'></a>

## 7. Credits and Acknowledgements

The web app is designed by following [Flask tutorials](https://www.youtube.com/playlist?list=PL-osiE80TeTs4UjLw5MM6OjgkjFeUxCYH) by Corey Schafer.

This github page is inspired by [this page](https://github.com/sanjeevai/disaster-response-pipeline).

<hr/> 

[Back to Table of Content](#index)