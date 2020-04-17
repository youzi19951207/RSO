import tensorflow as tf
import numpy as np


#===============Folder settings===================
CH=np.load("../data/#64_b64_100000_challenge_CNN_dataSet_8XOR.npy")
RSP=np.load('../data/#64_b64_100000_response_8XOR.npy')

print(np.shape(CH))

ans = []

for puf in range(1):
    NUM_TRAIN = 10000
    tf.reset_default_graph()  # Reset the default image
    graph = tf.Graph()  # New blank image
    with graph.as_default() as g:  # Use the newly created image as the default image
        with tf.Session(graph=g):
            x_train=CH[0:NUM_TRAIN*4]
            x_train=np.reshape(x_train,[-1,16*16])
            x_test=CH[90000*4:100000*4]
            x_test=np.reshape(x_test,[-1,16*16])


            y_train=np.array(RSP[puf,0:NUM_TRAIN]).reshape(NUM_TRAIN,1)
            y_test=np.array(RSP[puf,90000:100000]).reshape(10000,1)

            # one-hot coding
            y_train=tf.concat([1-y_train,y_train],1)
            y_test=tf.concat([1-y_test,y_test],1)


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

            # Define convolutional and pooling layers
            def conv2d(x,W):
                return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding = 'VALID')
            def max_pool_2x2(x):
                return tf.nn.max_pool(x, ksize = [1,2,2,1],strides = [1, 2, 2, 1], padding = 'SAME')


            #======================================Define input and output structure===================================================
            # Declare a placeholder, None means that the number of input pictures is uncertain, 28 * 28 picture resolution
            xs = tf.placeholder(tf.float32,[None, 16*16])
            # The category is 0-1, a total of 2 categories, corresponding to the output classification results
            ys = tf.placeholder(tf.float32, [None, 2])
            # x_image reshapes xs into a shape of 16 * 16 * 1. Because it is a gray picture, the channel is 1. As an input during training,
            # -1 represents the number of pictures

            x_image = tf.reshape(xs, [-1, 16, 16, 1])
            keep_prob = tf.placeholder(tf.float32)



            #================================Build a network, define the algorithm formula, which is the calculation of forward===========================================
            ## First layer convolution operation ##
            # The first and second parameters are worth the size of the convolution kernel, that is, patch, the third parameter is the number of image channels, and the 
            # fourth parameter is the number of convolution kernels, representing how many convolutional feature images will appear;
            W_conv1 = weight_variable([4, 4, 1, 64])
            # There is a corresponding offset for each convolution kernel
            b_conv1 = bias_variable([64])
            # The picture is multiplied by the convolution kernel, plus the paranoia amount, the convolution result is 16x16x64
            h_conv1 = tf.nn.sigmoid(tf.nn.conv2d(x_image, W_conv1, strides=[1, 2, 2, 1], padding='SAME')+b_conv1)
            # Pooling result 8x8x64 convolution result multiplied by pooled convolution kernel
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
            #  [n_samples, 6, 6, 64] ->> [n_samples, 6*6*64]
            h_conv2_flat = tf.reshape(h_conv2, [-1, 4*4*64])
            # Convolution operation, the result is 16 * 16, a single row multiplied by a single column is equal to 1 * 1 matrix,
            # matmul realizes the most basic matrix multiplication, different from tf.nn.conv2d traversal multiplication, it is automatically regarded as a front row vector and a back column vector
            h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
            h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

            ## Layer 3 fully connected operation ##
            W_fc2 = weight_variable([256, 64])
            b_fc2 = bias_variable([64])
            h_fc1_flat = tf.reshape(h_fc1_drop, [-1, 256])
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1_flat, W_fc2) + b_fc2)
            h_fc2_drop = tf.nn.dropout(h_fc2,keep_prob)

            ## Fourth layer output operation ##
            # Two-dimensional tensor, 1 * 1024 matrix convolution, a total of 10 convolutions, corresponding to the ys length we started is 10
            W_fc3 = weight_variable([64, 2])
            b_fc3 = bias_variable([2])
            # The final classification, the result is 1 * 1 * 2 softmax and sigmoid are based on logistic classification algorithm, one is 
            # multi-classification and one is two-classification
            y_conv = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)


            #=========================Define loss (minimum error probability), select optimization to optimize loss=============
            cross_entropy = -tf.reduce_sum(ys * tf.log(y_conv)) # Define cross entropy as loss function
            train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)# Call the optimizer to optimize, in fact, strive to minimize 

            #==========================Start data training and evaluation===============================
            correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(ys,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




            batch_size=1000
            i=0
            maxx = 0
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())  # Initialize variables
                for j in range(60):
                    for i in range(NUM_TRAIN//batch_size):
                        sess.run(train_step, feed_dict={xs:x_train[i*batch_size:(i+1)*batch_size], ys:y_train[i*batch_size:(i+1)*batch_size].eval(), keep_prob: 0.5})
                        if i%10==0:
                            train_accuracy,loss = sess.run([accuracy,cross_entropy], feed_dict={xs: x_test, ys: y_test.eval(), keep_prob: 1.0})
                            print("step %d,loss is %g,training accuracy %g"% (j*9+(i / 10 + 1),loss,train_accuracy))
                            maxx = max (maxx, train_accuracy)

                print("PUF",puf,"accuracy rate is：",maxx)
                ans.append(maxx)

print(ans)
print(sum(ans)/64.0)


