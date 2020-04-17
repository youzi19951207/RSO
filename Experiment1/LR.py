import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

NUM_KEY = 8

#===============Folder settings===================
CH=np.load("E:/data/#64_b64_100000_challenge_"+str(NUM_KEY)+"XOR.npy","r")
RSP=np.load("E:/data/#64_b64_100000_response_"+str(NUM_KEY)+"XOR.npy","r")
# CH=np.load("D:/data/#64_b64_100000_challenge.npy")
# RSP=np.load("D:/data/#64_b64_100000_response.npy").transpose()





NUM_Z = np.array([50,100,500,1000,5000,10000,50000,90000])
for NN in range(5,6):
    tf.reset_default_graph()

    graph = tf.Graph()
    with graph.as_default() as g:
        with tf.Session(graph=g):

            NUM_TRAIN = NUM_Z[NN]

            #==================parameter settings==================
            NUM_PUF=64
            NUM_CHBIT=64
            Num_step=10000
            fig=plt.figure()
            ax=plt.subplot(111)
            
            #=============Generate test set and training set==============

            #Process the challenges to extract features
            X_train=CH
            temp=1-2*X_train
            temp01 = np.array([np.prod(temp[:, i:], 1) for i in range(NUM_CHBIT)]).transpose()
            x_train=temp01[0:NUM_TRAIN]
            x_train=tf.Variable(x_train,dtype=tf.float32)
            x_test=temp01[90000:100000]
            x_test=tf.Variable(x_test,dtype=tf.float32)
            y_train=RSP[0:NUM_TRAIN]
            y_test=RSP[90000:100000]
            y_test=2*y_test-1

            # Set training parameters
            learning_rate = 0.1

            theta = tf.Variable(tf.zeros([NUM_CHBIT, NUM_PUF]))
            theta0 = tf.Variable(tf.zeros([1, NUM_PUF]))
            y = 1 / (1 + tf.exp(-(tf.matmul(x_train, theta) + theta0)))
            loss = tf.reduce_mean(- y_train * tf.log(y) - (1 - y_train) * tf.log(1 - y))
            train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


            init = tf.initialize_all_variables()
            sess = tf.Session()
            sess.run(init)

            # startime=time.time()
            A=[]
            B=[]

            for step in range(Num_step):
                sess.run(train)
                if step %50==0:
                    A.append(step)
                    #print('After %d times training, the cost is：'%(int(step)/50+1),sess.run(loss))
                    pred = tf.matmul(x_test, theta) + theta0
                    correct_prediction = tf.equal(tf.sign(pred), y_test)
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    a=sess.run(accuracy)
                    B.append(a)
                    print("Accuracy rate：")
                    #siz = np.shape(sess.run(pred))
                    #print(siz)
                    #print(sess.run(tf.reduce_sum(tf.cast(correct_prediction, tf.float32))))
                    print(a)
                # if (a>=0.95):
                #     break;
            #endtime=time.time()
            #time=endtime-startime
            #print('Characteristic value：', sess.run(theta).flatten(), sess.run(theta0).flatten())
            #print(np.shape(sess.run(theta)),np.shape(sess.run(theta0)))
            print(NUM_TRAIN,"The training set prediction accuracy is:",sess.run(accuracy))
            #print(time)
            #plt.plot(A,B,'c-',label='LR')
            #plt.xlabel("Number of iterations")
            #plt.ylabel("Prediction accuracy(%)")
            #plt.show()
