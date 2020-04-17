# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator,FixedFormatter
numKey=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,24,32])
CNN_A=np.array([99.46,71.42,66.78,64.75,63.46,63.11,62.01,60.14,58.61,58.24,57.47,56.71,56.43,56.40,56.17,55.66,55.03,53.94])
SVM_A=np.array([99.49,71.43,66.78,64.59,63.44,63.21,61.92,59.60,58.73,58.40,57.57,56.68,56.37,56.59,56.12,55.94,55.18,54.06])
LR_A=np.array( [99.55,71.44,66.80,64.75,63.49,62.90,61.85,59.70,58.58,58.42,57.32,56.63,56.43,56.40,56.20,55.72,55.02,53.93])
ANN_A=np.array([99.77,74.83,73.17,67.15,64.8,64.5,63.4,61.7,60.5,59.8,59.2,57.4,57.18,57.1,56.8,56.6,56.3,55.9])

# x2=2
# y2=71.3778

# x1=np.array([1000,5000,10000,50000,100000,200000,500000,1000000,1500000,2000000])
# y1=np.array([76,77.61,77.95,78.14,78.15,78.1795,78.280,78.34,78.42,78.43])
#
# x2=np.array([1000,5000,10000,50000,100000,200000,500000,1000000,1500000,2000000])
# y2=np.array([76,77.61,77.95,78.14,78.15,78.1795,78.280,78.34,78.42,78.43])
# x3=np.array([1000,2000000])
# y3=np.array([76,78.43])

a=np.array([50,100,500,1000,5000,10000,50000,100000,500000,1000000])
CNN=np.array([73.35,78.40,94.26,96.22,98.95,99.20,99.40,99.45,99.56,99.78])
SVM=np.array([72.73,80.38,92.55,95.20,98.40,98.85,99.35,99.49,99.79,99.88])
LR=np.array([ 72.54,79.06,93.64,95.57,98.51,98.83,99.43,99.55,99.76,99.80])
ANN=np.array([69.02,78.25,95.05,96.62,99.21,99.51,99.71,99.77,99.91,99.96])

fig=plt.figure()
ax=plt.subplot(111)
# plt.plot(x1,y1,"c:")
# plt.plot(x2, y2,"rX")
# plt.plot(x3,y3,"yX")
plt.plot(numKey,CNN_A,"rx-",linewidth=2.2,label="CNN")
plt.plot(numKey,SVM_A,"y*-.",linewidth=2.2,label="SVM")
plt.plot(numKey,LR_A,"bp-",linewidth=2.2,label="LR")
plt.plot(numKey,ANN_A,"k+-",linewidth=2.2,label="ANN")

# plt.annotate("Key=2",xy=(3.0,71.5),xytext=(9,77), fontsize=16,arrowprops=dict(facecolor='r',shrink=0.005))
# plt.xlabel("the number of keys")

# xmajorLocator=MultipleLocator(20000)
# xminorLocator=MultipleLocator(5000)
ymajorLocator=MultipleLocator(10)
yminorLocator=MultipleLocator(5)

# ax.xaxis.set_major_locator(xmajorLocator)
ax.yaxis.set_minor_locator(ymajorLocator)
# ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)

# plt.annotate(r'$(10^3,76)$', xy=(1000, 76), xytext=(700,75.4), fontsize=14,arrowprops=dict(facecolor='r',shrink=0.001))
# plt.annotate(r'$(2x10^6,78.4)$', xy=(2000000, 78.43), xytext=(200000,78.93), fontsize=14,arrowprops=dict(facecolor='r',shrink=0.001))
plt.ylim((50,100))
# ax.semilogx(a,LR)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.set_xlabel(..., fontsize=16)
ax.set_ylabel(..., fontsize=16)
# ax.legend(line1, ('CNN',))
# ax.legend(line2, ('SVM',))
# ax.legend(line3, ('LR',))
# ax.legend(line4, ('ANN',))
plt.legend(loc="upper right")



# plt.xlabel("the number of  training data")
plt.xlabel("the number of  keys")
plt.ylabel("prediction accuracy(%)")
plt.title("ML on a 64x64 Arbiter PUF",fontsize=16)
plt.show()



