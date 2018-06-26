from helpers import *
import random
import sys
rootdir = '/home/apo/Licenta/image_segmentation/raw_images/test'
background_set_path = '/home/apo/Licenta/image_segmentation/background_set_multi/test'

dimensions = [32, 48, 64, 80, 96, 112, 128]

#build 3000 image background set
for elem in os.listdir(rootdir):
	print(elem)
	if elem[-3:] == 'ppm':
			path = os.path.join(rootdir, elem)
			image = plt.imread(path)
			#pick a random dimension for the sliding window
			dim = random.choice(dimensions)
			windows = list(sliding_window(image, dim, (dim,dim)))
			#sample from the total number of windows to avoid balancing issues
			sample_windows = random.sample(windows, 65)
			for y,x,img in sample_windows:
				if img.shape[0] == img.shape[1]:
					img_name = os.path.join(background_set_path, elem[:-4])
					plt.imsave(fname=img_name+str(y)+'-'+str(x), arr=img)
