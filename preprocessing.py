# imports


# Import all the files
import os
from sklearn.model_selection import train_test_split as tts
import tensorflow as tf


files = []
X = []
y = []

for fname in os.listdir(os.path.join('./', 'landscapes')):
    files.append(os.path.join('./', 'landscapes', fname))

# Transform images into YUV and keep Y as X-set and UV as Y-set

def process_image(filename):
	image = tf.io.read_file(filename)
	image = tf.image.decode_jpeg(image) #0-255 integer
	image = tf.image.convert_image_dtype(image, tf.float32) #float 0-1
	image = tf.image.resize(image, [128,128]) #might get better results if 256x256 but we can experiment
	
	yuv_image = tf.image.rgb_to_yuv(image) #could do lab as well
	X_image = yuv_image[:,:,0] #just luminance 
	X_image = tf.expand_dims(X_image, axis=-1) #adding an extra dimension to the end of the vector because conv2d needs an extra dimension
	Y_image = yuv_image[:,:,1:] #colors

	return X_image, Y_image


for file in files:
    try:
        lum, uv = process_image(file)
        X.append(lum)
        y.append(uv)
    except:
        pass

# Split the data into train and test

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = tts(X_train, y_train, test_size=0.1, random_state=42)

print(X_train)
print(y_train)
