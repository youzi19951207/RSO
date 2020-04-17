import numpy as np

CH=np.load("D:/Data64_64/#64_b64_100000_challenge.npy")
CH=np.asarray(CH,int)
RSP=np.load("D:/Data64_64/#64_b64_100000_response.npy").transpose()
RSP=np.asarray(RSP,int)

print(CH)

Numbit=64
NUM_PUF=64
NumKey=8
NUM_CRP=100000

#============Generate key=======================
# key=np.random.randint(0,2,[NumKey,Numbit])
# print(key)
# np.save("D:/Data64_64/#8key",key)

key=np.load("D:/Data64_64/#8key.npy")
#=========================================




#===================obfuscate challenge and response==================
CH_XOR=np.zeros((NUM_CRP,Numbit))
RSP_XOR=np.zeros((NUM_CRP,NUM_PUF))

for i in range(NUM_CRP):
    # Randomly select a key from it for XOR
    key_choose = key[np.random.randint(0, NumKey)]
    for j in range(Numbit):
        a=CH[i,j]
        b=key_choose[j]
        c=(a&~b)|(~a&b)
        CH_XOR[i,j]=c
    for k in range(Numbit):
        a=RSP[i,k]
        b=key_choose[k]
        c=(a&~b)|(~a&b)
        RSP_XOR[i,k]=c
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
np.save("D:/Data64_64/#64_b64_100000_challenge_8XOR_samekey",CH_XOR)
np.save("D:/Data64_64/#64_b64_100000_response_8XOR_samekey",RSP_XOR)
