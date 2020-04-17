import numpy as np

CH=np.load("D:/万露的实验/data/#64_b64_100000_challenge.npy")
CH=np.asarray(CH,int)
RSP=np.load("D:/万露的实验/data/#64_b64_100000_response.npy").transpose()
RSP=np.asarray(RSP,int)

print(CH)


Numbit=64
NUM_PUF=64
NumKey=32




NUM_CRP=100000

#============产生key=======================
key=np.random.randint(0,2,[NumKey,Numbit])
#key=np.array([[1,1,0,1,0,1,0,0,1,0,0,1,0,0,1,0,0,0,1,0,1,0,0,0,0,1,1,1,1,1,0,0,1,1,0,0,0,1,0,0,0,0,0,1,1,1,1,1,0,0,1,1,0,0,1,1,0,0,1,0,0,0,0,1],
#             [0,1,1,0,1,0,1,1,0,1,1,0,1,1,0,1,1,1,0,1,0,1,1,1,1,0,0,0,0,0,1,1,0,0,1,1,1,0,1,1,1,1,1,0,0,0,0,0,1,1,0,0,1,1,0,0,1,1,0,1,1,1,1,0]])
print(key)
np.save("D:/万露的实验/data/#"+ str(NumKey) +"key",key)
#=========================================




#===================混淆challenge和response==================
CH_XOR=np.zeros((NUM_CRP,Numbit))
RSP_XOR=np.zeros((NUM_CRP,NUM_PUF))

for i in range(NUM_CRP):
    # 从中随机选择一个key进行异或
    key_choose = key[np.random.randint(0, NumKey)]
    for j in range(Numbit):
        a=CH[i,j]
        b=key_choose[j]
        c=(a&~b)|(~a&b)
        CH_XOR[i,j]=c

for i in range(NUM_CRP):
    # 从中随机选择一个key进行异或
    key_choose = key[np.random.randint(0, NumKey)]
    for j in range(Numbit):
        a=RSP[i,j]
        b=key_choose[j]
        c=(a&~b)|(~a&b)
        RSP_XOR[i,j]=c
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


print(CH_XOR)
print(np.shape(CH_XOR))

# print(CH_XOR_LFSR)
# print(np.shape(CH_XOR_LFSR))
np.save("D:/万露的实验/data/#64_b64_100000_challenge_"+ str(NumKey) +"XOR",CH_XOR)
np.save("D:/万露的实验/data/#64_b64_100000_response_"+ str(NumKey) +"XOR",RSP_XOR)
