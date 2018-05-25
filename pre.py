import numpy as np
from keras.models import load_model
from keras.preprocessing import image

names = ['egret', 'mandarin', 'owl', 'puffin', 'toucan', 'wood_duck']
model = load_model('first_blood.h5')

img_path = 'main_900.jpg'
img = image.load_img(img_path, target_size=(128,128))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)

preds = model.predict(x)
print(preds)
x = np.where(np.max(preds))[0][0]
print('The animal in this picture is:', names[x])
