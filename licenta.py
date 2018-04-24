import cv2
from keras.layers import Conv2D, Dropout, Dense, Activation, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import History
import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, pwd, pickle, glob
from skimage import transform
import csv

def get_username():
    return pwd.getpwuid( os.getuid() )[ 0 ]

def process_image(img):
	img = transform.resize(img, (150, 150))
	return img

def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.
    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    for c in range(0,43):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader, None) # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
            labels.append(row[7]) # the 8th column is the label
        gtFile.close()
    return images, labels

def createModel():
	'''create our cnn model using keras'''
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
	model.add(Activation('elu'))
	model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))

	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('elu'))
	model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))

	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('elu'))
	model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
	model.add(Dense(64))
	model.add(Activation('elu'))
	model.add(Dropout(0.2))

	model.add(Dense(43))
	model.add(Activation('softmax'))
	
	model.compile(loss='categorical_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'])
	return model


if __name__ == "__main__":
	if get_username()=='apo':
		root = '/home/apo/Licenta/GermanTrafficSignData'
	else:
		root = '/data'

	model = createModel()
	print(model.metrics_names)
	
	batch_size = 128
	
	trainPath = os.path.join(root, 'Training')
	testPath = os.path.join(root, 'Testing')
	
	x_train, y_train = readTrafficSigns(trainPath)
	x_test, y_test = readTrafficSigns(testPath)

	x_train = np.fromiter([process_image(img) for img in x_train])
	x_test = np.fromiter([process_image(img) for img in x_test])

	#Compute weight of each class
	#count number of images in each class
	imgCountDict = {}
	for elem in os.listdir(trainPath):
		if os.path.isdir(os.path.join(trainPath,elem)):
			newPath = os.path.join(trainPath, elem)
			imgCountDict[int(elem)] = len(os.listdir(newPath))-1 #-1 because each folder contains a csv file as well
	
	#now create a class_weights dictionary storing the weight of each class
	#using the most numerous class as a reference (most numerous class will have weight 1.0)
	reference = max(imgCountDict.values())
	class_weights = {x: float(imgCountDict[x]/reference) for x in imgCountDict.keys()}
	
	#create training and test generator

	
	print("Test generator class indices", test_generator.class_indices)
	
	#Fit generator on our model and evaluate it

	history = model.fit(x_train,
						y_train,
						shuffle=True,
						validation_split=0.2,
						validation_steps=800//batch_size,
						steps_per_epoch=2000//batch_size,
						epochs=80,
						class_weight=class_weights,
						)

	score = model.evaluate(x_test,
						   y_test,
						   steps=800//batch_size,
						   )
	
	print("Score={}\nHistory:{}".format(score, history.history))
	#Save model architecture
	with open("model_architecture.json", "w") as f:
		f.write(model.to_json())

	#Save model weights
	model.save_weights('model_weights.h5')
