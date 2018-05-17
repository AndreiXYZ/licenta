import cv2
from keras.layers import Conv2D, Dropout, Dense, Activation, Flatten, MaxPooling2D, BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import History, TerminateOnNaN, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam, SGD
from keras.models import load_model
from keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, pwd, pickle, glob
from skimage import transform

def get_username():
    return pwd.getpwuid( os.getuid() )[ 0 ]

def process_image(img_path):
	img = plt.imread(img_path)
	img = transform.resize(img, (150, 150, 3))
	return img

def createModel():
	'''create our cnn model using keras'''
	model = Sequential()
	model.add(Conv2D(32, (3, 3), strides=(1,1), input_shape=(32, 32, 3)))
	model.add(Activation('elu'))
	model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))

	model.add(Conv2D(64, (3, 3), strides=(1,1)))
	model.add(Activation('elu'))
	model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))

	model.add(Conv2D(128, (3, 3), strides=(1,1)))
	model.add(Activation('elu'))
	
	model.add(Flatten())

	model.add(Dense(1024, kernel_regularizer=l2(0.0001)))
	model.add(Activation('elu'))
	model.add(Dropout(0.3))

	model.add(Dense(43))
	model.add(Activation('softmax'))
	
	model.compile(loss='categorical_crossentropy',
				  optimizer=Adam(lr=0.003),
				  metrics=['accuracy'])
	return model


if __name__ == "__main__":
	if get_username()=='apo':
		root = '/home/apo/Licenta/German-Augmented'
	else:
		root = '/data'

	model = createModel()
	print(model.metrics_names)
	
	batch_size = 256
	
	trainPath = os.path.join(root, 'Training')
	testPath = os.path.join(root, 'Testing')
	valPath = os.path.join(root, 'Validation')

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
	
	#create training, validation and test generator
	datagen = ImageDataGenerator(rescale=1./255,
								 rotation_range=10,
        						 width_shift_range=0.2,
        						 height_shift_range=0.2,
        						 shear_range=0.2,
        						 zoom_range=0.2,
        						 fill_mode='nearest')

	training_generator = datagen.flow_from_directory(
						trainPath,
						target_size=(32,32),
						batch_size=batch_size,
						class_mode='categorical'
						)


	val_datagen = ImageDataGenerator(rescale=1./255)
	validation_generator = val_datagen.flow_from_directory(
									valPath,
									target_size=(32,32),
									batch_size=batch_size,
									class_mode='categorical')

	test_datagen = ImageDataGenerator(rescale=1./255)

	test_generator = test_datagen.flow_from_directory(
									 testPath,
									 target_size=(32,32),
									 batch_size=batch_size,
									 class_mode='categorical')

	#create a list of callbacks for our model to use
	terminate_nan = TerminateOnNaN()
	checkpoint_loss = ModelCheckpoint(filepath='/output/best_model_loss.h5',
								 monitor='val_loss',
								 save_best_only=True,
								 save_weights_only=False,
								 mode='auto',
								 period=1)

	checkpoint_acc = ModelCheckpoint(filepath='/output/best_model_acc.h5',
								 monitor='val_acc',
								 save_best_only=True,
								 save_weights_only=False,
								 mode='auto',
								 period=1)

	reduce_lr = ReduceLROnPlateau(monitor='loss',
										  factor=0.8,
										  patience=3,
										  mode='auto',
										  min_lr=0.0001,
										  epsilon=0.0001)

	callbacks = [terminate_nan, checkpoint_loss, checkpoint_acc, reduce_lr]
	
	#fit generator on our model and evaluate it
	history = model.fit_generator(
						training_generator,
						validation_data=validation_generator,
						shuffle=True,
						steps_per_epoch=25000//batch_size,
						validation_steps=1300//batch_size,
						epochs=300,
						class_weight=class_weights,
						#use_multiprocessing=True,
						callbacks=callbacks
						#workers=12
						)

	score = model.evaluate_generator(
						test_generator,
						steps=12500//batch_size
						#use_multiprocessing=True
						#workers=12
						)
	
	print("Score on last model={}".format(score))

	bestModelLoss = load_model('/output/best_model_loss.h5')
	bestModelAcc = load_model('/output/best_model_acc.h5')

	best_model_loss_score = bestModelLoss.evaluate_generator(
						test_generator,
						steps=12500//batch_size,
						#use_multiprocessing=True
						)

	best_model_acc_score = bestModelAcc.evaluate_generator(
						test_generator,
						steps=12500//batch_size,
						#use_multiprocessing=True
						)

	print("Score of best model (by loss)={}".format(best_model_loss_score))
	print("Score of best model (by acc)={}".format(best_model_acc_score))
	#Save history obj for visualisation
	print("History={}".format(history.history))
	#Save model architecture
	bestModelLoss.save("/output/best_model_loss.h5")
	bestModelAcc.save("/output/best_model_acc.h5")
	model.save("/output/model.h5")
