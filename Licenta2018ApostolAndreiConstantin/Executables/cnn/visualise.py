import pandas as pd
import os
import matplotlib.pyplot as plt

if __name__=="__main__":
	root = '/home/apo/Licenta/GermanTrafficSignData/Training'
	os.chdir(root)
	countDict = {}
	for elem in os.listdir(root):
		countDict[int(elem)] = len(os.listdir(elem))

	plt.bar(countDict.keys(), countDict.values(), color='b')
	plt.show()