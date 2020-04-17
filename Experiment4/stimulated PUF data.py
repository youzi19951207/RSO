import numpy as np

NUM_RSPBIT = 128
NUM_CHBIT = 128
NUM_CRP = 100000
NUM_SELECT = 3    # 2 3 4
NUM_DATA = 2 ** NUM_SELECT
NUM_PUF = NUM_DATA + NUM_SELECT



#===========存储每一个stage的4个延迟参数===========
mean = 10
stdev = 0.05
alpha = 0
Data = np.random.normal(mean, stdev, (NUM_PUF*NUM_CHBIT, NUM_CHBIT * 4))


#===========产生Challenges============
C=np.random.randint(0,2,(NUM_CRP,NUM_CHBIT))


#=============产生Response====================
R=np.zeros((NUM_CRP,NUM_RSPBIT))

cc = [[1, 0, 0, -1], [-1, 0, 0, 1], [0, 1, -1, 0], [0, -1, 1, 0]]
cz = [1, -1, -1, 1]

def Arbiter(c, data):
    '''
        :param c: challenge
        :param d: delays
        :return: parameter vector and response
    '''
    # # Add noise
    # if alpha != 0:
    #     noise = np.random.normal(0, alpha * stdev, (1, NUM_CHBIT*4))
    #     data = d + noise[0, :]
    # else:
    #     data = d
    # transfer input

    flag = 0
    x = []
    y = []
    for i in c:
        x.append(cc[2 * i + flag])
        y.append(cz[2 * i + flag])
        if i == 1:
            flag = 1 - flag  # i=1 Path flipping
    sum = 0.0  # Sum of delay differences
    for i in range(4):
        for j in range(NUM_CHBIT):
            sum += x[j][i] * data[i * NUM_CHBIT + j]
    r = (1+np.sign(sum))/2
    return int(r)


for i in range(NUM_CRP):
    if i%1000==0:
        print(i)
    for j in range(NUM_RSPBIT):
        S = 0
        c = C[i, :]
        for k in range(NUM_SELECT):
            r = Arbiter(c, Data[k + j * NUM_PUF, :])
            S = 2 * S + r
        R[i, j] = Arbiter(c, Data[S + NUM_SELECT + j * NUM_PUF, :])

#===============将CRP用数组CRP数组保存============
# CRP=np.zeros((NUM_CRP,NUM_CHBIT+NUM_PUF))
# for i in range(NUM_CRP):
#     for j in range(NUM_CHBIT):
#         CRP[i,j]=CH[i,j]
#     CRP[i,NUM_CHBIT:]=RSP[:,i]



#===========将文件保存至文件夹===========

# for f in range(NUM_PUF):
#     file_crp='G:/Data/newData32/#32_b32_10000.crp'
#     fw=open(file_crp,"w+")
#     for c in range(NUM_CRP):
#         for b in range(NUM_CHBIT):
#             print(CH[c,b])
#             fw.write(str(CH[c,b])+",")
#         fw.write(str(RSP[f,c]))
#         fw.write('\n')
#     fw.close()

#==================二进制保存===========
np.save('./data/CH_MPUF_' + str(NUM_CHBIT) + '_' + str(NUM_RSPBIT) + '_' + str(NUM_CRP), C)
np.save('./data/R_MPUF_' + str(NUM_CHBIT) + '_' + str(NUM_RSPBIT) + '_' + str(NUM_CRP), R)


