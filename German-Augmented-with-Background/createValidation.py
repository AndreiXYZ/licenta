import os
import random
import shutil
import glob
import sys
root = '/home/apo/Licenta/German-Augmented/Training'


for cl in os.listdir(root):
	images = os.listdir(os.path.join(root,cl))
	img = random.choice(images)
	track_name = img.split('_')[0]
	for track_image in glob.glob(os.path.join(root, cl) + "/" + track_name + "*.ppm"):
		newPath = track_image.replace('Training', 'Validation')
		shutil.move(track_image, newPath)
