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
from keras.utils import plot_model

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

model = createModel()
plot_model(model, to_file='best_model_loss.png', show_shapes=True)