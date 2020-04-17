import cma
import numpy as np


'''----------------------------------My init---------------------------------------------'''
CH = np.load("F:/万露/万露的实验2/CMA/#1_b64_100000_challenge.npy")
# RSP=np.load('G:/Data/Data64_error/#1_b64_100000_response_10%.npy').transpose()
RSP = np.load('F:/万露/万露的实验2/CMA/#1_b64_100000_response.npy')
NUM_CHBIT = 64
train_num = 50000
X_train = CH
temp = 1 - 2 * X_train
temp01 = np.array([np.prod(temp[:, i:], 1) for i in range(NUM_CHBIT)]).transpose()
x_train = temp01[0:train_num]
x_test = temp01[90000:100000]
y_train = RSP.transpose()[0:train_num]
y_test = RSP.transpose()[90000:100000]
y_test = 2 * y_test - 1

'''--------------------------------------------------------------------------------------'''


w = 65*[0.1]
es = cma.CMAEvolutionStrategy(w, 10)
print(es.result)

es.optimize(cma.ff.Myfit1)

