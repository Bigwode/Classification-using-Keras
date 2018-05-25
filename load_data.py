# coding:utf-8
import keras
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import numpy as np
import random
import os
import cv2

CLASS_NUM = 6
names = ['egr', 'man', 'owl', 'puf', 'tou', 'wod']


def load_data(img_width, img_height, path):
    print("[INFO] loading images...")
    data = []
    labels = []
    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(path)))
    random.seed(42)
    random.shuffle(imagePaths)
    # loop over the input images
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (img_width, img_height))
        image = img_to_array(image)
        data.append(image)

        # extract the class label from the image path and update the labels list
        name = imagePath.split(os.path.sep)[-1][:3]
        label = int(names.index(name))
        labels.append(label)


    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # convert the labels from integers to vectors
    labels = to_categorical(labels, num_classes=CLASS_NUM)
    return data, labels