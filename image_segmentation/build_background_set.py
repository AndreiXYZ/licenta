from helpers import *

rootdir = '/home/apo/Licenta/image_segmentation/FullIJCNN2013'
background_set_path = '/home/apo/Licenta/image_segmentation/background_set'
(winW, winH) = (128, 128)

while len(os.listdir(background_set_path)) < 3000:
	for elem in os.listdir(rootdir):
		if elem[-3:] == 'ppm':
				path = os.path.join(rootdir, elem)
				image = plt.imread(path)
				for y,x,img in sliding_window(image, 64, (winW, winH)):
					if img.shape[0] == img.shape[1]:
						img_name = os.path.join(background_set_path, elem[:-4])
						plt.imsave(fname=img_name+str(y)+'-'+str(x), arr=img)
