import tensorflow as tf
import numpy as np


#===============文件夹设置===================
CH=np.load("D:/万露的实验/data/#64_b64_100000_challenge_CNN.npy")
RSP=np.load('D:/万露的实验/data/#64_b64_100000_response.npy').transpose()
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

# # one-hot 编码
# y_train=tf.concat(1,[1-y_train,y_train])
# y_test=tf.concat(1,[1-y_test,y_test])


#========================================函数声明部分=====================================
#初始化权重和偏向
def weight_variable(shape):
    # 正态分布，标准差为0.1，默认最大为1，最小为-1，均值为0
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    # 创建一个结构为shape矩阵也可以说是数组shape声明其行列，初始化所有值为0.1
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

#定义卷积层和池化层
def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding = 'VALID')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1],strides = [1, 2, 2, 1], padding = 'SAME')


#======================================定义输入输出结构===================================================
# 声明一个占位符，None表示输入图片的数量不定，28*28图片分辨率
xs = tf.placeholder(tf.float32,[None, 16*16])

# 类别是0-1总共2个类别，对应输出分类结果
ys = tf.placeholder(tf.float32, [None, NUM_PUF])

# x_image又把xs reshape成了16*16*1的形状，因为是灰色图片，所以通道是1.作为训练时的input，-1代表图片数量不定
x_image = tf.reshape(xs, [-1, 16, 16, 1])
keep_prob = tf.placeholder(tf.float32)



#================================搭建网络,定义算法公式，也就是forward时的计算===========================================
## 第一层卷积操作 ##
# 第一二参数值得卷积核尺寸大小，即patch，第三个参数是图像通道数，第四个参数是卷积核的数目，代表会出现多少个卷积特征图像;
W_conv1 = weight_variable([4, 4, 1, 64])
# 对于每一个卷积核都有一个对应的偏置量
b_conv1 = bias_variable([64])
# 图片乘以卷积核，并加上偏执量，卷积结果16x16x64
h_conv1 = tf.nn.sigmoid(tf.nn.conv2d(x_image, W_conv1, strides=[1, 2, 2, 1], padding='SAME')+b_conv1)
#池化结果8x8x64 卷积结果乘以池化卷积核
#h_pool1 = tf.nn.max_pool(h_conv1, ksize = [1,2,2,1],strides = [1, 2, 2, 1], padding = 'SAME')

## 第二层卷积操作 ##
# 64通道卷积，卷积出64个特征
W_conv2 = weight_variable([4,4,64,64])
# 64个偏执数据
b_conv2 = bias_variable([64])
# 注意h_pool1是上一层的池化结果，#卷积结果8x8x64
h_conv2 = tf.nn.sigmoid(tf.nn.conv2d(h_conv1, W_conv2, strides=[1,2,2,1], padding='SAME')+b_conv2)
#赤化结果4x4x64
# h_pool2= tf.nn.max_pool(h_conv2, ksize = [1,2,2,1],strides = [1, 2, 2, 1], padding = 'SAME')

## 第三层全连接操作 ##
# 二维张量，第一个参数8*8*64的patch，也可以认为是只有一行4*4*64个数据的卷积，第二个参数代表卷积个数共16x16个
W_fc1 = weight_variable([4*4*64, 16*16])
# 256个偏执数据
b_fc1 = bias_variable([256])
# 将第二层卷积结果reshape成只有一行6*6*64个数据
#  [n_samples, 6, 6, 64] ->> [n_samples, 6*6*64]
h_conv2_flat = tf.reshape(h_conv2, [-1, 4*4*64])
# 卷积操作，结果是16*16，单行乘以单列等于1*1矩阵，matmul实现最基本的矩阵相乘，不同于tf.nn.conv2d的遍历相乘，自动认为是前行向量后列向量
h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

## 第三层全连接操作 ##
W_fc2 = weight_variable([256, 64])
b_fc2 = bias_variable([1,64])
h_fc1_flat = tf.reshape(h_fc1_drop, [-1, 256])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_flat, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2,keep_prob)

## 第四层输出操作 ##
# 二维张量，1*1024矩阵卷积，共10个卷积，对应我们开始的ys长度为10
W_fc3 = weight_variable([64,64])
b_fc3 = bias_variable([1,64])
# 最后的分类，结果为1*1*2 softmax和sigmoid都是基于logistic分类算法，一个是多分类一个是二分类
y_pred=tf.matmul(h_fc2_drop, W_fc3) + b_fc3
# y_conv = tf.nn.softmax(y_pred)
#
#
# #=========================定义loss(最小误差概率)，选定优化优化loss=============
# cross_entropy = -tf.reduce_sum(ys * tf.log(y_conv)) # 定义交叉熵为loss函数
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)# 调用优化器优化，其实就是通过喂数据争取cross_entropy最小化

y = tf.sigmoid(y_pred)
train_step = tf.reduce_mean(-y_train*tf.log(y)-(1-y_train)*tf.log(1-y))
train = tf.train.GradientDescentOptimizer(0.1).minimize(train_step)

#==========================开始数据训练以及评测===============================
correct_prediction = tf.equal(tf.sign(y_pred), y_test)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer()) #初始化变量



for i in range(9000):
    sess.run(train, feed_dict={xs:x_train, ys:y_train, keep_prob: 1.0})
    if i%10==0:
        train_accuracy= sess.run(accuracy, feed_dict={xs: x_test, ys: y_test, keep_prob: 1.0})
        print("training accuracy %g"% (train_accuracy))



