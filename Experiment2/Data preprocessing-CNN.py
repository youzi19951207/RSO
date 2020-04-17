import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time


#===============文件夹设置===================
CH=np.load("../Data/#64_b64_100000_challenge_8XOR.npy")
RSP=np.load("../Data/#64_b64_100000_response.npy")

print(np.shape(CH))
print(CH)
print(np.shape(RSP))




#==================参数设置==================
NUM_PUF=1
NUM_CHBIT=64
NUM_CRP=100000

#=============生成测试集和训练集===============

#将challenges做处理提取出特征
X_train=CH
temp=1-2*X_train
temp01 = np.array([np.prod(temp[:, i:], 1) for i in range(NUM_CHBIT)]).transpose()
temp02=-temp01



#==================数据转换==================

newChallenge=np.zeros([len(CH)*4,len(CH[0])])

#将数据进行扩展成64x2x2的数据
for i in range(len(CH)):
    newChallenge[i*4]=temp01[i]
    newChallenge[i*4+1]=temp02[i]
    newChallenge[i*4+2,0:32]=temp01[i,32:64]
    newChallenge[i*4+2,32:64]=temp01[i,0:32]
    newChallenge[i*4+3,0:32]=temp02[i,32:64]
    newChallenge[i*4+3,32:64]=temp02[i,0:32]

# print(temp01[0])
# print(temp02[0])
# print(newChallenge[0:4])
np.save("../Data/#64_b64_100000_challenge_CNN_dataSet_8XOR",newChallenge)




