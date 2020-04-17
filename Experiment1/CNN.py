import tensorflow as tf
import numpy as np


#===============Folder settings===================
CH=np.load("D:/data/#64_b64_100000_challenge_CNN.npy")
RSP=np.load('D:/data/#64_b64_100000_response.npy').transpose()
print(np.shape(CH))
print(np.shape(RSP))

NUM_PUF=64
NUM_CHBIT=64

x_train=CH[0:90*4]
x_train=np.reshape(x_train,[-1,16*16])

x_test=CH[90*4:100*4]
x_test=np.reshape(x_test,[-1,16*16])

y_train=RSP[0:90]

y_test=RSP[90:100]
y_test=2*y_test-1

# # one-hot coding
# y_train=tf.concat(1,[1-y_train,y_train])
# y_test=tf.concat(1,[1-y_test,y_test])


#========================================Function declaration section=====================================
#Initialization weights and bias
def weight_variable(shape):
    # Normal distribution, standard deviation is 0.1, default maximum is 1, minimum is -1, mean is 0
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    # Create a structure for the shape matrix can also be said that the array shape declares its ranks, initialize all values to 0.1
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

#Define convolutional and pooling layers
def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding = 'VALID')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1],strides = [1, 2, 2, 1], padding = 'SAME')


#======================================Define input and output structure===================================================
# Declare a placeholder, None means that the number of input pictures is uncertain, 28 * 28 picture resolution
xs = tf.placeholder(tf.float32,[None, 16*16])

# The category is 0-1, a total of 2 categories, corresponding to the output classification results
ys = tf.placeholder(tf.float32, [None, NUM_PUF])

# x_image reshapes xs into a shape of 16 * 16 * 1. Because it is a gray picture, the channel is 1. As an input during training,
#-1 represents the number of pictures
x_image = tf.reshape(xs, [-1, 16, 16, 1])
keep_prob = tf.placeholder(tf.float32)



#================================Build a network, define the algorithm formula, which is the calculation of forward===========================================
## First layer convolution operation ##
# The first and second parameters are worth the size of the convolution kernel, that is, patch, the third parameter is the 
#number of image channels, and the fourth parameter is the number of convolution kernels, representing how many convolutional feature images will appear;
W_conv1 = weight_variable([4, 4, 1, 64])
# There is a corresponding offset for each convolution kernel
b_conv1 = bias_variable([64])
# The picture is multiplied by the convolution kernel, plus the paranoia amount, the convolution result is 16x16x64
h_conv1 = tf.nn.sigmoid(tf.nn.conv2d(x_image, W_conv1, strides=[1, 2, 2, 1], padding='SAME')+b_conv1)
#Pooling result 8x8x64 convolution result multiplied by pooled convolution kernel
#h_pool1 = tf.nn.max_pool(h_conv1, ksize = [1,2,2,1],strides = [1, 2, 2, 1], padding = 'SAME')

## Second layer convolution operation ##
# 64-channel convolution with 64 features
W_conv2 = weight_variable([4,4,64,64])
# 64 bias data
b_conv2 = bias_variable([64])
# Note that h_pool1 is the pooling result of the previous layer, and the convolution result is 8x8x64
h_conv2 = tf.nn.sigmoid(tf.nn.conv2d(h_conv1, W_conv2, strides=[1,2,2,1], padding='SAME')+b_conv2)
# Pooling result 4x4x64
# h_pool2= tf.nn.max_pool(h_conv2, ksize = [1,2,2,1],strides = [1, 2, 2, 1], padding = 'SAME')

## Layer 3 fully connected operation ##
# Two-dimensional tensor, the patch with the first parameter 8 * 8 * 64 can also be considered as a convolution with 
# only one row of 4 * 4 * 64 data, and the second parameter represents the total number of convolutions 16x16
W_fc1 = weight_variable([4*4*64, 16*16])
# 256 bias data
b_fc1 = bias_variable([256])
# Reshape the second layer convolution result into only one row of 6 * 6 * 64 data
# [n_samples, 6, 6, 64] ->> [n_samples, 6*6*64]
h_conv2_flat = tf.reshape(h_conv2, [-1, 4*4*64])
# Convolution operation, the result is 16 * 16, a single row multiplied by a single column is equal to 1 * 1 matrix, 
# matmul realizes the most basic matrix multiplication, different from tf.nn.conv2d traversal multiplication, it is 
# automatically regarded as a front row vector and a back column vector
h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

## Layer 3 fully connected operation ##
W_fc2 = weight_variable([256, 64])
b_fc2 = bias_variable([1,64])
h_fc1_flat = tf.reshape(h_fc1_drop, [-1, 256])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_flat, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2,keep_prob)

## Fourth layer output operation ##
# Two-dimensional tensor, 1 * 1024 matrix convolution, a total of 10 convolutions, corresponding to the ys length we started is 10
W_fc3 = weight_variable([64,64])
b_fc3 = bias_variable([1,64])
# The final classification, the result is 1 * 1 * 2 softmax and sigmoid are based on logistic classification algorithm, one is 
# multi-classification and one is two-classification
y_pred=tf.matmul(h_fc2_drop, W_fc3) + b_fc3
# y_conv = tf.nn.softmax(y_pred)
#
#
# #=========================Define loss (minimum error probability), select optimization to optimize loss=============
# cross_entropy = -tf.reduce_sum(ys * tf.log(y_conv)) # Define cross entropy as loss function
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

y = tf.sigmoid(y_pred)
train_step = tf.reduce_mean(-y_train*tf.log(y)-(1-y_train)*tf.log(1-y))
train = tf.train.GradientDescentOptimizer(0.1).minimize(train_step)

#==========================Start data training and evaluation===============================
correct_prediction = tf.equal(tf.sign(y_pred), y_test)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer()) #Initialize variables



for i in range(9000):
    sess.run(train, feed_dict={xs:x_train, ys:y_train, keep_prob: 1.0})
    if i%10==0:
        train_accuracy= sess.run(accuracy, feed_dict={xs: x_test, ys: y_test, keep_prob: 1.0})
        print("training accuracy %g"% (train_accuracy))



