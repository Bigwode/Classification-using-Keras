# coding:utf-8
import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras import optimizers
import numpy as np
import argparse
import sys
import mulNet
import load_data

# dimensions of our images.
img_width, img_height = 128, 128

nb_train_samples = 1126
# nb_validation_samples = 60
epochs = 100
batch_size = 32

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


model = mulNet.build(img_width, img_height)
# print(model.summary())


def train(X_train, X_test, y_train, y_test):

    opt = optimizers.RMSprop(lr=0.001 ,decay=1e-6)
    model.compile(loss='categorical_crossentropy', # 多分类
                  optimizer=opt,  # 'rmsprop'
                  # loss_weights=[0.1, 0.9],
                  metrics=['accuracy'])

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        # rotation_range=30,
        # rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        fill_mode = "nearest"
    )

    train_generator = train_datagen.flow(X_train, y_train, batch_size=32, shuffle=True, seed=None)
    val_generator = ImageDataGenerator().flow(X_test, y_test, batch_size=32, shuffle=True, seed=None)

    # train_generator = train_datagen.flow_from_directory(
    #     train_data_dir,
    #     target_size=(img_width, img_height),
    #     batch_size=batch_size,
    #     class_mode='categorical')

    # val_generator = val_datagen.flow_from_directory(
    #     val_data_dir,
    #     target_size=(img_width, img_height),
    #     batch_size=batch_size,
    #     class_mode='categorical')

    model.fit_generator(
        train_generator,
        validation_data=val_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs)


    model.save('first_blood.h5')


if __name__=='__main__':

    arg = argparse.ArgumentParser(description='Process the input_output path.')
    arg.add_argument("-path", "--dataset_path", default='./birds/train',
                     help="path to input dataset_train")
    # arg.add_argument("-dtrain", "--dataset_train", default='./birds/train',
    #                  help="path to input dataset_train")
    # arg.add_argument("-dval", "--dataset_val", default='./birds/val',
    #                  help="path to input dataset_val")
    args = arg.parse_args()

    # train_data_dir = vars(args)['dataset_train']  # './birds/train'
    # val_data_dir = vars(args)["dataset_val"]  # './birds/val'
    train_data_dir = vars(args)['dataset_path']
    train_data, train_labels = load_data.load_data(img_width, img_height, train_data_dir)

    X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size = 0.3, random_state = 42)

    train(X_train, X_test, y_train, y_test)
