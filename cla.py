# coding:utf-8
import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from keras import backend as K
from keras import optimizers
import numpy as np
import argparse
import sys
import mulNet
import load_data

# dimensions of our images.
img_width, img_height = 224, 224

nb_train_samples = 1126
# nb_validation_samples = 60
epochs = 10
batch_size = 32

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


# model = mulNet.build_normal(img_width, img_height)
base_model, model = mulNet.build_vgg_raw(img_width, img_height)


# print(model.summary())


def train(X_train, X_test, y_train, y_test):

    # opt = optimizers.RMSprop(lr=0.001 ,decay=1e-6)
    # model.compile(loss='categorical_crossentropy', # 多分类
    #               optimizer=opt,  # 'rmsprop'
    #               # loss_weights=[0.1, 0.9],
    #               metrics=['accuracy'])

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

    print('训练顶层分类器')
    for layer in base_model.layers:
        layer.trainable = False

    opt = optimizers.RMSprop(lr=0.001 ,decay=1e-6)
    model.compile(loss='categorical_crossentropy', # 多分类
                  optimizer=opt,  # 'rmsprop'
                  # loss_weights=[0.1, 0.9],
                  metrics=['accuracy'])

    history_t1 = model.fit_generator(
        train_generator,
        validation_data=val_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs)

    print('对顶层分类器fine-tune')
    for layer in model.layers[:11]:
        layer.trainable = False
    for layer in model.layers[11:]:
        layer.trainable = True

    opt = optimizers.SGD(lr=0.0001, momentum=0.9)
    model.compile(loss='categorical_crossentropy',  # 多分类
                  optimizer=opt,  # 'rmsprop'
                  # loss_weights=[0.1, 0.9],
                  metrics=['accuracy'])

    history_ft = model.fit_generator(
        train_generator,
        validation_data=val_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs)


    model.save('first_blood.h5')
    plot_training(history_ft)


def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.show()



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
