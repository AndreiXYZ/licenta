import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
	return 1/(1+np.exp(-x))

def relu(x):
	return max(0,x)

def leaky_relu(x):
	if x<0:
		return 0.02*x
	return x

def elu(x):
	if x<0:
		return np.exp(x)-1
	return x
f, axarr = plt.subplots(1,4)

x = np.linspace(-6,6,100)

axarr[0].grid()
axarr[0].set_title('Sigmoid')
axarr[0].plot(x,sigmoid(x))

axarr[1].grid()
axarr[1].set_title('ReLU')
axarr[1].plot(x,list(map(relu,x)))

axarr[2].grid()
axarr[2].set_title('Leaky ReLU (alpha=0.02)')
axarr[2].plot(x,list(map(leaky_relu,x)))

axarr[3].grid()
axarr[3].set_title('ELU (alpha=1.0)')
axarr[3].plot(x, list(map(elu,x)))
plt.show()