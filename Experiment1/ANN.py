import numpy as np
import tensorflow as tf
import time

NUM_KEY = 8

#===============文件夹设置===================
CH=np.load("E:/万露的实验/data/#64_b64_100000_challenge_"+str(NUM_KEY)+"XOR.npy")
RSP=np.load("E:/万露的实验/data/#64_b64_100000_response_"+str(NUM_KEY)+"XOR.npy").transpose()
# CH=np.load("D:/万露的实验/data/#64_b64_100000_challenge.npy")
# RSP=np.load("D:/万露的实验/data/#64_b64_100000_response.npy").transpose()


#==================参数设置==================
NUM_CHBIT=64
NUM_PUF=64
NUM_MID=64
NUM_TRAIN = 90000
print(np.shape(CH))
print(np.shape(RSP))

#=============生成测试集和训练集===============

#将challenges做处理提取出特征
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



# 添加层
def add_layer(inputs,Weights,biases,in_size, out_size,keep_prob=1.0, activation_function=None):
    # add one more layer and return the output of this layer
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
        outputs=tf.nn.dropout(outputs,keep_prob)
    return outputs

# 2.定义节点准备接收数据
#输入层
NUM_l1 =64
NUM_l2 =96
NUM_l3 =64
Weights = tf.Variable(tf.random_normal([NUM_CHBIT, NUM_l1],stddev=0.1))
biases = tf.Variable(tf.zeros([NUM_l1]))

# define placeholder for inputs to network
x = tf.placeholder(tf.float32, [None, NUM_CHBIT])
y_ = tf.placeholder(tf.float32, [None,NUM_PUF])
keep_prob=tf.placeholder(tf.float32)


# 3.定义神经层：隐藏层和预测层
# add hidden layer 输入值是 xs，在隐藏层有 30 个神经元
l1 = add_layer(x,Weights,biases, NUM_CHBIT, NUM_l1,keep_prob, activation_function=tf.nn.sigmoid)

Weights2 = tf.Variable(tf.random_normal([NUM_l1, NUM_l2],stddev=0.1))
biases2 = tf.Variable(tf.zeros([NUM_l2]))

l2 = add_layer(l1,Weights2,biases2, NUM_l1, NUM_l2,keep_prob, activation_function=tf.nn.sigmoid)

Weights3 = tf.Variable(tf.random_normal([NUM_l2, NUM_l3],stddev=0.1))
biases3 = tf.Variable(tf.zeros([NUM_l3]))

l3 = add_layer(l2,Weights3,biases3, NUM_l2, NUM_l3,keep_prob, activation_function=tf.nn.sigmoid)

#输出层
w=tf.Variable(tf.zeros([NUM_l1,NUM_PUF]))
b=tf.Variable(tf.zeros([NUM_PUF]))
y=tf.nn.sigmoid(tf.matmul(l1,w)+b)




# 4.定义 loss 表达式
# the error between prediciton and real data
#loss = tf.reduce_mean(- y_ * tf.log(y) - (1 - y_) * tf.log(1 - y))
loss = tf.reduce_mean(tf.square(y - y_))
# 5.选择 optimizer 使 loss 达到最小
# 这一行定义了用什么方式去减少 loss，学习率是 0.01

train = tf.train.AdamOptimizer(0.005).minimize(loss)


# important step 对所有变量进行初始化
init = tf.global_variables_initializer()
sess = tf.Session()
# 上面定义的都没有运算，直到 sess.run 才会开始运算
sess.run(init)


# 迭代 1000 次学习，sess.run optimizer
for i in range(700000):
    # training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数
    sess.run(train, feed_dict={x: x_train, y_: y_train,keep_prob:0.75})
    if i % 20 == 0:
        # to see the step improvement
        print("此时的loss：",sess.run(loss,feed_dict={x: x_train, y_: y_train,keep_prob:1.0}))
        l1=add_layer(x_test,Weights,biases, NUM_CHBIT, NUM_l1,keep_prob, activation_function=tf.nn.sigmoid)
        l2 = add_layer(l1, Weights2, biases2, NUM_l1, NUM_l2, keep_prob, activation_function=tf.nn.sigmoid)
        l3 = add_layer(l2, Weights3, biases3, NUM_l2, NUM_l3, keep_prob, activation_function=tf.nn.sigmoid)
        pred=tf.matmul(l1,w)+b
        correct_prediction = tf.equal(tf.sign(pred), y_test)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("此时准确率为：",sess.run(accuracy,feed_dict={keep_prob:1.0}))



acuracy=sess.run(accuracy,feed_dict={keep_prob:1.0})
print("最终的准确率为：",acuracy)

