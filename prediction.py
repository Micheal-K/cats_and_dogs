import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

model_dir = '/Users/Wangyaxi/Project/cats-and-dogs/model/c_d_3.h5'
model = tf.keras.models.load_model(model_dir)

# Read test images
img_path = '/Users/Wangyaxi/Project/cats-and-dogs/my_predict/4.png'
img = image.load_img(img_path, target_size=(150, 150))

# Convert images to numpy arrays and normalize
x = image.img_to_array(img)
x = x / 255.0
x = np.expand_dims(x, axis=0)

#  predictions
preds = model.predict(x)
print(preds)
# Print prediction results
if preds[0] > 0.5:
    print('It is a Dog')
else:
    print('It is a Cat')
