import numpy as np

NUM_RSPBIT = 128
NUM_CHBIT = 128
NUM_CRP = 100000
NUM_SELECT = 3    # 2 3 4
NUM_DATA = 2 ** NUM_SELECT
NUM_PUF = NUM_DATA + NUM_SELECT
NUM_KEY = 8

CH=np.load('./data/CH_MPUF_' + str(NUM_CHBIT) + '_' + str(NUM_RSPBIT) + '_' + str(NUM_CRP) + '.npy')
CH=np.asarray(CH,int)
RSP=np.load('./data/R_MPUF_' + str(NUM_CHBIT) + '_' + str(NUM_RSPBIT) + '_' + str(NUM_CRP) + '.npy')
RSP=np.asarray(RSP,int)

#============产生key=======================
key=np.random.randint(0,2,[NUM_KEY,NUM_RSPBIT])
#key=np.array([[1,1,0,1,0,1,0,0,1,0,0,1,0,0,1,0,0,0,1,0,1,0,0,0,0,1,1,1,1,1,0,0,1,1,0,0,0,1,0,0,0,0,0,1,1,1,1,1,0,0,1,1,0,0,1,1,0,0,1,0,0,0,0,1],
#             [0,1,1,0,1,0,1,1,0,1,1,0,1,1,0,1,1,1,0,1,0,1,1,1,1,0,0,0,0,0,1,1,0,0,1,1,1,0,1,1,1,1,1,0,0,0,0,0,1,1,0,0,1,1,0,0,1,1,0,1,1,1,1,0]])
# # np.save("../data/#"+ str(NumKey) +"key",key)
#=========================================


#===================混淆challenge和response==================
key1 = np.array([key[np.random.randint(0, NUM_KEY)] for i in range(NUM_CRP)])
CH_XOR = np.bitwise_xor(CH, key1)
key2 = np.array([key[np.random.randint(0, NUM_KEY)] for j in range(NUM_CRP)])
RSP_XOR = np.bitwise_xor(RSP, key2)

#============================================================
# CH_XOR_LFSR=np.zeros((100000*64,Numbit))
#
# for m in range(100000):
#     CH_XOR_LFSR[m*64]=CH_XOR[m]
#     for n in range(1,Numbit,1):
#             CH_XOR_LFSR[m+n,0:n]=CH_XOR[m,Numbit-n:Numbit]
#             CH_XOR_LFSR[m+n,n:Numbit]=CH_XOR[m,0:Numbit-n]
#
#
# np.save("D:/Data64_LFSR/#1_b64_100000_challenge.npy_2XOR",CH_XOR)
# np.save("D:/Data64_LFSR/#1_b64_100000_challenge.npy_2XOR_LFSR",CH_XOR_LFSR)


print(np.shape(CH_XOR))
print(np.shape(RSP_XOR))

# print(CH_XOR_LFSR)
# print(np.shape(CH_XOR_LFSR))
np.save('./data/CH_DMOS_' + str(NUM_CHBIT) + '_' + str(NUM_RSPBIT) + '_' + str(NUM_CRP) + '_' + str(NUM_KEY), CH_XOR)
np.save('./data/R_DMOS_' + str(NUM_CHBIT) + '_' + str(NUM_RSPBIT) + '_' + str(NUM_CRP) + '_' + str(NUM_KEY), RSP_XOR)

