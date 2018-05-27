import numpy as np
from keras.models import load_model
from keras.preprocessing import image

names = ['egret', 'mandarin', 'owl', 'puffin', 'toucan', 'wood_duck']
model = load_model('first_blood.h5')

img_path = 'oowwll.jpg'
img = image.load_img(img_path, target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
preds = model.predict(x)
print(preds[0])
pre_list = preds[0].tolist()
x = pre_list.index(max(pre_list))
print('The animal in this picture is:', names[x])
