# coding:utf-8
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Input, Embedding, Dropout, Flatten, Dense
from keras.models import Model



def build(img_width, img_height):
    input_image = Input(shape=(img_width, img_height, 3))
    x1 = (Conv2D(32, (3, 3), activation='relu'))(input_image)
    x1 = MaxPooling2D(pool_size=(2, 2))(x1)

    x2 = Conv2D(64, (3, 3), activation='relu')(x1)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)

    x31 = Conv2D(128, (3, 3), activation='relu')(x2)
    x31 = MaxPooling2D(pool_size=(2, 2))(x31)

    x41 = GlobalAveragePooling2D()(x31)
    # Flatten()(x31)
    x51 = Dense(64, activation='relu')(x41)
    # x61 = Dropout(0.5)(x51)
    # prediction1 = Dense(6, activation='softmax')(x61) # 6分类


    x32 = Conv2D(256, (3, 3), activation='relu')(x2)
    x32 = MaxPooling2D(pool_size=(2, 2))(x32)

    x42 = Conv2D(256, (3, 3), activation='relu')(x32)
    x42 = MaxPooling2D(pool_size=(2, 2))(x42)

    x52 = GlobalAveragePooling2D()(x42)
    x62 = Dense(64, activation='relu')(x52)
    merged_vector = keras.layers.concatenate([x51, x62], axis=-1)
    x72 = Dropout(0.5)(merged_vector)
    prediction = Dense(6, activation='softmax')(x72) # 6分类

    model = Model(inputs=input_image, outputs=prediction)
    return model
