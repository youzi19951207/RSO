

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


#===============文件夹设置===================
CH=np.load("../data/#64_b64_100000_challenge_8XOR.npy")
RSP=np.load('../data/#64_b64_100000_response_8XOR.npy')

print(np.shape(CH))
print(CH)
print(np.shape(RSP))


#==================参数设置==================
NUM_PUF=1
NUM_CHBIT=64
NUM_TRAIN = 5000

ans = []
for puf in range(64):
#=============生成测试集和训练集==============
#将challenges做处理提取出特征

    tf.reset_default_graph()  # 重置默认图
    graph = tf.Graph()        # 新建空白图
    with graph.as_default() as g:   # 将新建的图作为默认图
        with tf.Session(graph=g):
            X_train=CH
            temp=1-2*X_train
            temp01 = np.array([np.prod(temp[:, i:], 1) for i in range(NUM_CHBIT)]).transpose()
            x_train=temp01[0:NUM_TRAIN]
            #x_train=tf.Variable(x_train,dtype=tf.float32)
            x_test=temp01[90000:100000]
            x_test=tf.Variable(x_test,dtype=tf.float32)

            y_train=np.array(RSP[puf,0:NUM_TRAIN]).reshape(NUM_TRAIN,1)
            y_train=2*y_train-1
            y_test=np.array(RSP[puf,90000:100000]).reshape(10000,1)
            y_test=2*y_test-1


            lr = 0.01
            epoch = 5000

            x = tf.placeholder(tf.float32, [None, 64])
            y = tf.placeholder(tf.float32, [None, 1])

            W = tf.Variable(tf.random_normal([64,1]), dtype=tf.float32, name='w')
            b = tf.Variable(tf.random_normal(shape=[1, 1]), dtype=tf.float32, name='b')

            y_pred1 = tf.add(tf.matmul(x, W), b)
            l2_norm = tf.reduce_sum(tf.square(W))
            y_pred = tf.sign(y_pred1)
            alpha = tf.constant([0.01])
            classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(y_pred1, y))))

            loss = tf.add(classification_term, tf.multiply(alpha, l2_norm)/2)


            optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

            #tf.summary.scalar('loss', loss)
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                merged = tf.summary.merge_all()
                writer = tf.summary.FileWriter('nlogs/', sess.graph)
                sess.run(init)
                maxx = 0
                for i in range(epoch):
                    _, loss_,W_,b_= sess.run([optimizer,loss,W,b], feed_dict={x: x_train, y:y_train})
                    y_pred_,y_pred1_, w = sess.run([y_pred,y_pred1,W], feed_dict={x: x_train, y:y_train})
                    if i%50 ==0:
                        loss_ = sess.run(loss, feed_dict={x: x_train, y:y_train})
                        #print('经过%d轮训练cost为：' % (int(i) / 50 + 1),loss_)
                        #result= sess.run(merged, feed_dict={x: x_train, y:y_train})
                        #writer.add_summary(result, i)
                        pred = tf.matmul(x_test, W) +b
                        correct_prediction = tf.equal(tf.sign(pred), y_test)
                        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                        a = sess.run(accuracy)
                        #print(a)
                        maxx = max(maxx , a )
                ans.append(maxx)
            print("PUF",puf,"的准确率：",maxx)

print(ans)
print(sum(ans)/64.0)

