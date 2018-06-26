import numpy as np
import matplotlib.pyplot as plt
import os

ims = []
f, axarr = plt.subplots(4,5)

cols = 5
rows = 4

for elem in os.listdir():
	if elem[-3:] == 'ppm':
		ims.append(plt.imread(elem))

print(len(ims))
for row in range(4):
	for col in range(5):
		axarr[row,col].imshow(ims[row+col*4])
		axarr[row,col].axis('off')

plt.show()