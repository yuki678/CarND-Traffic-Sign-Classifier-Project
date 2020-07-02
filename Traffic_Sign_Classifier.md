# Traffic Sign Recognition

---

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./images/sample_train_images1.png "Sample Input 1"
[image2]: ./images/sample_train_images2.png "Sample Input 2"
[image3]: ./images/train_labels1.png "Input Labels 1"
[image4]: ./images/train_labels2.png "Input Labels 2"
[image5]: ./images/preprocessed_images.png "Preprocessed image"
[image6]: ./images/augmented.png "Augmented image"
[image7]: ./images/learning.png "Learning Curve"

[image8]: ./images/feature_map1.png "Feature Map 1"
[image9]: ./images/feature_map2.png "Feature Map 2"
[image10]: ./images/feature_map3.png "Feature Map 3"

Here is a link to my [project code](https://github.com/yuki678/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

---

## Data Set Summary & Exploration

### 1. A basic summary of the data set

I used the pandas library to calculate summary statistics of the traffic signs data set:
```python
n_train = X_train.shape[0]
n_validation = X_valid.shape[0]
n_test = X_test.shape[0]
image_shape = X_train.shape[1:]
n_classes = signnames.shape[0]

```
* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

### 2. An exploratory visualization of the dataset

Here is an exploratory visualization of the data set. Please find sample images for all the 43 classes in the notebook.
![alt text][image1]
:
![alt text][image2]

Training data labes are distributed to all classes in a similar way between train, validation adn test sets, while there are some imbalances. This imbalance will be rectified later.
![alt text][image3]
![alt text][image4]

## Design and Test a Model Architecture

### 1. Preprocess

I first convert the images to grayscale because it tends to perform better than color image for this type of classification of signs. Then, equalize the histogram and normalize the image for the better optimization of gradient decent.
Here is an example of a traffic sign image before and after the preprocessing.

![alt text][image5]

I decided to generate additional data to increase the number of training dataset to avoid overfitting.
To add more data to the the data set, I used the following techniques.

```python
def rotate(image, angle=15):
    angle = np.random.randint(-angle, angle)
    M = cv2.getRotationMatrix2D((16, 16), angle, 1)
    return cv2.warpAffine(src=image, M=M, dsize=(32, 32))

def translate(image, pixels=2):
    tx = np.random.choice(range(-pixels, pixels))
    ty = np.random.choice(range(-pixels, pixels))
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(src=image, M=M, dsize=(32, 32))

def random_bright(image):
    eff = 0.5 + np.random.random()
    return image * eff

def normalize_im(image):
    mini, maxi = np.min(image), np.max(image)
    return (image - mini) / (maxi - mini) * 2 - 1

def generate(images, count):
    generated = []
    while True:
        for image in images:
            if len(generated) == count:
                return generated
            image = random_bright(image)
            image = rotate(image)
            image = translate(image)
            image = normalize_im(image)
            generated.append(np.expand_dims(image, axis=2))
```

Here is an example of an original image and an augmented image:
![alt text][image6]

After the augmentation, the number of training dataset became 215,000.

### 2. The final model architecture 

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x12 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x24 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 16x16x36 	|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 16x16x48 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x48 		    		|
| Fully connected		| 5136 to 512 neurons               			|
| RELU					|												|
| Dropout   	      	| 0.5 (50%) dropout          		    		|
| Fully connected		| 512 to 256 neurons               		    	|
| RELU					|												|
| Dropout   	      	| 0.5 (50%) dropout          		    		|
| Softmax				|           									| 


### 3. Training
To train the model, I used the following hyperparameters and 
```python
batch_size = 32
keep_prob = .5
epochs = 100
patience = 3

logits, conv1, conv2 = traffic_sign_model(x, kp)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_operation = optimizer.minimize(loss_operation)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()
```

Also, I used exponential decay for the learning late so that it gets smaller as approaching the optimal point.
```python
# Exponential decaying learning rate
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(
    learning_rate=0.0005,
    global_step=global_step,
    decay_steps=ceil(n_train / batch_size),  # Decay every epoch
    decay_rate=0.95,
    staircase=True)
```

### 4. Improving the validation set accuracy to be at least 0.93.

 Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.996
* test set accuracy of 0.977

I initially tried without data augmentation and iterative approach. The model was two layers without dropout but it overfits and did not perform the validation set well.
Then, I added those features to the model and training to reach the validation set accuracy of 0.997.

I also tried different values for the hyperparameters and decided to use exponential decay for the learning rate as it performed better than a fixed rate.
I actually first struggled to decide what layers and how many to stack. Tried many different combinations and finally decided the model referring many different articles.

Now the model fits to validation set well as well as the train set.

![alt text][image7]

The result on the test set ensures the model is robust to make a prediction on unseen data.

## Test a Model on New Images

### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five actual German traffic signs photos that I found on the web:

<img src="./data/web_test/germany_sign1.png" alt="traffic sign 1" width="300"/>
<img src="./data/web_test/germany_sign2.jpg" alt="traffic sign 1" width="300"/>
<img src="./data/web_test/germany_sign3.jpg" alt="traffic sign 1" width="300"/>
<img src="./data/web_test/germany_sign4.jpg" alt="traffic sign 1" width="300"/>
<img src="./data/web_test/germany_sign5.jpg" alt="traffic sign 1" width="300"/>

I cropped the images, applied the preprocess and fed into the trained model for the prediction.

### 2. The model's predictions on these new traffic signs

Here are the results of the prediction:

| Image			                            | Prediction	        	| 
|:-----------------------------------------:|:-------------------------:| 
| 40: Roundabout mandatory      		    | 40: Roundabout mandatory 	| 
| 32: End of all speed and passing limits   | 40: Roundabout mandatory 	|
| 1: Speed limit (30km/h)				    | 1: Speed limit (30km/h)	|
| 13: Yield	      		                    | 13: Yield					|
| 12: Priority road		                	| 12: Priority road      	|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 0.977.

### 3. The softmax probabilities for each prediction

The resultant top 5 softmax probabilities using `tf.nn.top_k(softmax, k=5)` was the following.

```python
Test accuracy: 80.000%
TopKV2(values=array([[  1.00000000e+00,   3.29862455e-35,   8.31373689e-36,
          3.58961839e-37,   0.00000000e+00],
       [  9.85343754e-01,   1.35716302e-02,   1.08460302e-03,
          1.02317677e-09,   6.26324201e-11],
       [  1.00000000e+00,   1.47379966e-08,   3.42104095e-10,
          8.81711718e-11,   2.52603906e-12],
       [  1.00000000e+00,   1.95015076e-17,   8.83404467e-20,
          3.41077990e-21,   2.03948408e-21],
       [  1.00000000e+00,   2.11156877e-18,   7.05565003e-19,
          1.73972665e-19,   7.26577291e-20]], dtype=float32), indices=array([[40,  1, 37, 12,  0],
       [ 1, 40,  4,  0,  7],
       [ 1, 40,  2,  4,  5],
       [13, 15, 12, 35,  9],
       [12, 38, 40, 35, 13]], dtype=int32))
```
This shows that first, third, fourth and fifth pictures are classified correctly with almost 100% confidence.
The model could not classify the second picture correctly because the picture is not the same as images in the training data. Obviously, the model cannot predict a new class which is not in the training data. The picture is the end of speed 30km/h, which is a combination of end of speed and speed limit of 30km/h. In fact, the model predicted this as Speed limit (30km/h) with 98% confidence and if we have more training data with 'the end of xxx' sign, the model could probably consider it as a candidate class and we could make a model to use top 2 combination when the picture contains two class features at the same level.

## (Optional) Visualizing the Neural Network
### 1. Feature Map
I computed the feature map using the following code. The resultant charts follows.
```python
def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1, plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
            
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    ix = int(np.random.random() * X_test_n.shape[0])
    random_image = np.expand_dims(X_test_n[ix], axis=0)
    
    plt.imshow(X_test[ix])
    plt.title("Input: " + signnames[y_test[ix]], fontsize=20, y=1.02)
    plt.show()

    outputFeatureMap(random_image, conv1, plt_num=1)
    plt.suptitle("Feature Map of the First Convolutional Layer", fontsize=30, y=0.92)
    plt.show()
    
    outputFeatureMap(random_image, conv2, plt_num=2)
    plt.suptitle("Feature Map of the Second Convolutional Layer", fontsize=30, y=0.92)
    plt.show()
```
![alt text][image8]
![alt text][image9]
![alt text][image10]

The first layer captures the edges and shapes in the image. Then the second convolutional layer seems to capture the patterns contained in the image.