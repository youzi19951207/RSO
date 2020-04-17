import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

ans_f = []

for key in range(0,32,2):
 
#===============Folder settings===================
    CH=np.load("../data/#64_b64_100000_challenge_8XOR.npy")
    #RSP=np.load('G:/Data/Data64_error/#1_b64_100000_response_10%.npy').transpose()
    RSP=np.load('../data/#64_b64_100000_response_8XOR.npy')

    #==================parameter settings==================

    ans = []
    NUM_P = 64
    for puf in range(NUM_P):

        NUM_PUF=1
        NUM_CHBIT=64
        NUM_TRAIN = 90000

        #=============Generate test set and training set==============
        tf.reset_default_graph()  # Reset the default image
        graph = tf.Graph()        # New blank image
        with graph.as_default() as g:   # Use the newly created image as the default image
            with tf.Session(graph=g):
                # Process the challenges to extract features
                X_train=CH
                temp=1-2*X_train
                temp01 = np.array([np.prod(temp[:, i:], 1) for i in range(NUM_CHBIT)]).transpose()
                x_train=temp01[0:NUM_TRAIN]
                x_train=tf.Variable(x_train,dtype=tf.float32)
                x_test=temp01[90000:1000000]
                x_test=tf.Variable(x_test,dtype=tf.float32)
                y_train=np.array(RSP[puf,0:NUM_TRAIN]).reshape(NUM_TRAIN,1)
                y_test=np.array(RSP[puf,90000:100000]).reshape(10000,1)
                y_test=2*y_test-1

                #print(np.shape(x_train))
                #print(np.shape(y_train))
                #Set training parameters

                learning_rate = 0.01

                theta = tf.Variable(tf.zeros([NUM_CHBIT, NUM_PUF]))
                theta0 = tf.Variable(tf.zeros([1, NUM_PUF]))
                y = 1 / (1 + tf.exp(-(tf.matmul(x_train, theta) + theta0)))
                loss = tf.reduce_mean(- y_train * tf.log(y) - (1 - y_train) * tf.log(1 - y))
                train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

                init = tf.global_variables_initializer()
                sess = tf.Session()
                sess.run(init)
                maxx = 0
                for step in range(100000):
                    sess.run(train)
                    if step %500==0:

                        pred = tf.matmul(x_test, theta) + theta0
                        correct_prediction = tf.equal(tf.sign(pred), y_test)
                        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                        a=sess.run(accuracy)
                        maxx = max(maxx , a)
                        if a < 0.1:
                            break
                        print('After %d training times, the accuracy rate is：' % (int(step) / 500+ 1), a)
                    # if (a>=0.95):
                    #     break;
                #print('Characteristic value is：', sess.run(theta).flatten(), sess.run(theta0).flatten())
                #print(np.shape(sess.run(theta)),np.shape(sess.run(theta0)))
                print("%d training set prediction accuracy rate is:",maxx)
                ans.append(maxx)
    print(ans)
    print(sum(ans)/NUM_P)
    ans_f.append(sum(ans)/NUM_P)


np.save("E:/data_XOR",ans_f)
