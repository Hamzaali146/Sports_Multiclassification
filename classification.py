# import dataset from kaggle include kaggle
# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/

# !kaggle datasets download -d samuelcortinhas/sports-balls-multiclass-image-classification

import cv2 as cv
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout
from tensorflow.keras.applications import MobileNet
import zipfile
zip_ref = zipfile.ZipFile('/content/sports-balls-multiclass-image-classification.zip', 'r')
zip_ref.extractall('/content')
zip_ref.close()

#lets generate train and evaluate dataset
train_ds= keras.utils.image_dataset_from_directory(
    directory= '/content/train',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(256,256)
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory= '/content/test',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(256,256)
)

 
 # Here we are doing Normalization
def process(image,label):
    image = tf.cast(image/255. ,tf.float32)
    return image,label

train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)
input_shape = (256, 256, 3)

base_model = MobileNet(input_shape=input_shape, 
                       include_top=False,
                       weights='imagenet', 
                       classes=15, 
                       classifier_activation="softmax")

for layer in base_model.layers:
    layer.trainable = False

#CNn Model
 
model= Sequential()
model.add(base_model)


model.add(Conv2D(32, kernel_size=(3,3),padding='valid',activation='relu',input_shape=(224,224,3)))
model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(64, kernel_size=(3,3),padding='valid',activation='relu'))
model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(128, kernel_size=(3,3),padding='valid',activation='relu'))
model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(15,activation='softmax'))
model.summary()


from keras.optimizers import Adam

# Define the optimizer with the desired learning rate
optimizer = Adam(learning_rate=0.001)

model.compile(optimizer='sgd',
              loss=tf.keras.losses.CategoricalCrossentropy())

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history=model.fit(train_ds,epochs=20, validation_data=validation_ds) 

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'],color='red',label='train')
plt.plot(history.history['val_accuracy'],color='blue',label='validation')
plt.legend()
plt.show()


test_img = cv.imread('/content/cricket-ball.jpg')
plt.imshow(test_img)

test_img.shape
test_img = cv.resize(test_img,(256,256))
test_input = test_img.reshape((1,256,256,3))

model.predict(test_input)
test_img1 = cv.imread('/content/football.jpg')
plt.imshow(test_img1)

test_img1 = cv.resize(test_img1,(256,256))
test_input1 = test_img1.reshape((1,256,256,3))

model.predict(test_input1)