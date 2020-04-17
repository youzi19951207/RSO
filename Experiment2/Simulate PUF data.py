import numpy as np

NUM_PUF=64
NUM_CHBIT=64
NUM_CRP=100000

FILE_DATA="F:/data/#64_b64.txt"




#===========Store 4 delay parameters for each stage===========
DATA_p=np.zeros((NUM_PUF,NUM_CHBIT))
DATA_q=np.zeros((NUM_PUF,NUM_CHBIT))
DATA_r=np.zeros((NUM_PUF,NUM_CHBIT))
DATA_s=np.zeros((NUM_PUF,NUM_CHBIT))

f=open(FILE_DATA,'r')
for i in range(NUM_PUF):
    for j in range(NUM_CHBIT):
        DATA_p[i,j]=float(f.readline().strip())
        DATA_q[i,j]=float(f.readline().strip())
        DATA_r[i,j]=float(f.readline().strip())
        DATA_s[i,j]=float(f.readline().strip())


#===========Generate Challenges============
CH=np.random.randint(0,2,(NUM_CRP,NUM_CHBIT))

# CH=np.load("D:/Data64_LFSR/#1_b64_10000_challenge.npy")

#=============Generate Response====================
RSP=np.zeros((NUM_PUF,NUM_CRP))

for f in range(NUM_PUF):
    for c in range(NUM_CRP):
        pa=0
        pb=0
        flag=0
        for b in range (NUM_CHBIT-1,-1,-1):
            if (flag==0):
                if(CH[c,b]==0):
                    pa+=DATA_p[f,b]
                    pb+=DATA_q[f,b]
                else:
                    pa+=DATA_s[f,b]
                    pb+=DATA_r[f,b]
                    flag=1-flag
            else:
                if(CH[c,b]==0):
                    pa+=DATA_q[f,b]
                    pb+=DATA_p[f,b]
                else:
                    pa+=DATA_r[f,b]
                    pb+=DATA_s[f,b]
                    flag=1-flag
        if(pa>pb):
            RSP[f,c]=0
            RSP[f,c]=int(RSP[f,c])
        else:
            RSP[f,c]=1
            RSP[f,c]=int(RSP[f, c])

#===============Save CRP with array CRP array============
# CRP=np.zeros((NUM_CRP,NUM_CHBIT+NUM_PUF))
# for i in range(NUM_CRP):
#     for j in range(NUM_CHBIT):
#         CRP[i,j]=CH[i,j]
#     CRP[i,NUM_CHBIT:]=RSP[:,i]



#===========Save file to folder===========

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

#==================Binary save===========
print(CH)
np.save("F:/data/#64_b64_100000_challenge",CH)
print(RSP)
np.save("F:/data/#64_b64_100000_response",RSP)
print(np.shape(CH))
print(np.shape(RSP))


