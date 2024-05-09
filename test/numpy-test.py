import numpy as np

tensor = np.arange(12)
print(tensor)
tensor = tensor.reshape((1,6,2))
print(tensor)
tensor = tensor.reshape((1,2,6))
print(tensor)
tensor1 = np.arange(12)
tensor1 = tensor1.reshape((1,6,2))
tensor11 = tensor1[:,0:2,:]
tensor12 = tensor1[:,2:4,:]
tensor13 = tensor1[:,4:6,:]
tensor1 = np.concatenate((tensor11,tensor12,tensor13),axis=2)
print(tensor1)