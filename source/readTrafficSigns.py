# The German Traffic Sign Recognition Benchmark
#
# sample code for reading the traffic sign images and the
# corresponding labels
#
# example:
#            
# trainImages, trainLabels = readTrafficSigns('GTSRB/Training')
# print len(trainLabels), len(trainImages)
# plt.imshow(trainImages[42])
# plt.show()
#
# have fun, Christian

import matplotlib.pyplot as plt
import csv
import os
import pickle
import numpy as np
# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 

def augmentImage(image, datagen, iterations=10, dir=".", ):
    img = load_img(image)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=dir, save_prefix="augmented", save_format="ppm"):
        i += 1
        if i>iterations:
            break

def process_image(img_path):
    img = plt.imread(img_path)
    img = transform.resize(img, (150,150))
    return img

def get_class_dictionary(path):
    for elem in os.listdir(path):
        newPath = os.path.join(path, elem)
        imgCountDict[int(elem)] = len(os.listdir(newPath))-1 #-1 because each folder contains a csv file as well

def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = np.array([], dtype=np.float32) # images
    labels = np.array([], dtype=np.float32) # corresponding labels
    # loop over all 42 classes
    for c in range(0,43):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader, None) # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            np.append(images, plt.imread(prefix + row[0])) # the 1th column is the filename
            np.append(labels, row[7]) # the 8th column is the label
        gtFile.close()
    return images, labels


rootpath = '/home/apo/Licenta/GTSRB'
trainpath = os.path.join(rootpath, 'Training')
testpath = os.path.join(rootpath, 'Testing')
x_train, y_train = readTrafficSigns(trainpath)
print(x_train.shape, len(x_train))
print(y_train.shape, len(y_train))
x_test, y_test = readTrafficSigns(testpath)

