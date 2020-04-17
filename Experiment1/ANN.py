import numpy as np
import tensorflow as tf
import time

NUM_KEY = 8

#===============Folder settings===================
CH=np.load("E:/data/#64_b64_100000_challenge_"+str(NUM_KEY)+"XOR.npy")
RSP=np.load("E:/data/#64_b64_100000_response_"+str(NUM_KEY)+"XOR.npy").transpose()
# CH=np.load("D:/data/#64_b64_100000_challenge.npy")
# RSP=np.load("D:/data/#64_b64_100000_response.npy").transpose()


#==================Parameter settings==================
NUM_CHBIT=64
NUM_PUF=64
NUM_MID=64
NUM_TRAIN = 90000
print(np.shape(CH))
print(np.shape(RSP))

#=============Generate test set and training set===============

#Process the challenges to extract features
X_train=CH
temp=1-2*X_train
temp01 = np.array([np.prod(temp[:, i:], 1) for i in range(NUM_CHBIT)]).transpose()
x_train=temp01[0:NUM_TRAIN]
#x_train=tf.Variable(x_train,dtype=tf.float32)
x_test=temp01[90000:100000]
x_test=tf.Variable(x_test,dtype=tf.float32)

y_train=RSP[0:NUM_TRAIN]
y_test=RSP[90000:100000]
y_test=2*y_test-1
print(np.shape(x_train))



# 1.Add layer
def add_layer(inputs,Weights,biases,in_size, out_size,keep_prob=1.0, activation_function=None):
    # add one more layer and return the output of this layer
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
        outputs=tf.nn.dropout(outputs,keep_prob)
    return outputs

# 2.Define the node ready to receive data
# input layer
NUM_l1 =64
NUM_l2 =96
NUM_l3 =64
Weights = tf.Variable(tf.random_normal([NUM_CHBIT, NUM_l1],stddev=0.1))
biases = tf.Variable(tf.zeros([NUM_l1]))

# define placeholder for inputs to network
x = tf.placeholder(tf.float32, [None, NUM_CHBIT])
y_ = tf.placeholder(tf.float32, [None,NUM_PUF])
keep_prob=tf.placeholder(tf.float32)


# 3.Define the neural layer: hidden layer and prediction layer
# add hidden layer. The input value is xs and there are 30 neurons in the hidden layer
l1 = add_layer(x,Weights,biases, NUM_CHBIT, NUM_l1,keep_prob, activation_function=tf.nn.sigmoid)

Weights2 = tf.Variable(tf.random_normal([NUM_l1, NUM_l2],stddev=0.1))
biases2 = tf.Variable(tf.zeros([NUM_l2]))

l2 = add_layer(l1,Weights2,biases2, NUM_l1, NUM_l2,keep_prob, activation_function=tf.nn.sigmoid)

Weights3 = tf.Variable(tf.random_normal([NUM_l2, NUM_l3],stddev=0.1))
biases3 = tf.Variable(tf.zeros([NUM_l3]))

l3 = add_layer(l2,Weights3,biases3, NUM_l2, NUM_l3,keep_prob, activation_function=tf.nn.sigmoid)

# out layer
w=tf.Variable(tf.zeros([NUM_l1,NUM_PUF]))
b=tf.Variable(tf.zeros([NUM_PUF]))
y=tf.nn.sigmoid(tf.matmul(l1,w)+b)

# 4.Define loss expression
# the error between prediciton and real data
#loss = tf.reduce_mean(- y_ * tf.log(y) - (1 - y_) * tf.log(1 - y))
loss = tf.reduce_mean(tf.square(y - y_))

# 5.Choose optimizer to minimize loss
# This line defines the way to reduce loss, the learning rate is 0.01

train = tf.train.AdamOptimizer(0.005).minimize(loss)

# important step Initialize all variables
init = tf.global_variables_initializer()
sess = tf.Session()
# None of the operations defined above will start operations until sess.run
sess.run(init)


# 1000 iterations，sess.run optimizer
for i in range(700000):
    # Both training train_step and loss are operations defined by placeholder, so here we need to use feed to pass in the parameters
    sess.run(train, feed_dict={x: x_train, y_: y_train,keep_prob:0.75})
    if i % 20 == 0:
        # to see the step improvement
        print("Loss at this time：",sess.run(loss,feed_dict={x: x_train, y_: y_train,keep_prob:1.0}))
        l1=add_layer(x_test,Weights,biases, NUM_CHBIT, NUM_l1,keep_prob, activation_function=tf.nn.sigmoid)
        l2 = add_layer(l1, Weights2, biases2, NUM_l1, NUM_l2, keep_prob, activation_function=tf.nn.sigmoid)
        l3 = add_layer(l2, Weights3, biases3, NUM_l2, NUM_l3, keep_prob, activation_function=tf.nn.sigmoid)
        pred=tf.matmul(l1,w)+b
        correct_prediction = tf.equal(tf.sign(pred), y_test)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("The accuracy rate is：",sess.run(accuracy,feed_dict={keep_prob:1.0}))



acuracy=sess.run(accuracy,feed_dict={keep_prob:1.0})
print("The final accuracy is：",acuracy)

