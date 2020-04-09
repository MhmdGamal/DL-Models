#Cat/Dog Image Classification (CNN)

#Check The Images To Initialize your Categories List
test_imgs,test_labels=next(test_data)
import matplotlib.pyplot as plt
plt.imshow(test_imgs[0])
print(test_labels[0])
#figured that the cat's image in label 0
categories=['cat','dog']


#importing Libraries
from keras.models import Sequential
from keras.layers import Conv2D,Dense,MaxPool2D,Flatten
from keras.callbacks import ModelCheckpoint

checkpoint=ModelCheckpoint('weight.hdf5',monitor='val_loss',save_best_only=True)
#initializing the CNN
model=Sequential()
#Convolution Layer
model.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=(64,64,3),padding='same'))
#Pooling
model.add(MaxPool2D(pool_size=(2,2)))
#Convolution Layer
model.add(Conv2D(32,kernel_size=3,activation='relu',padding='same'))
#Pooling
model.add(MaxPool2D(pool_size=(2,2)))
#Fully Connection
model.add(Flatten())
model.add(Dense(64,activation='relu'))
#output Layer
model.add(Dense(1,activation='sigmoid'))
#Compiling The Model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)
training_data=train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64,64),
                                                batch_size=32,
                                                class_mode='binary')
test_data=test_datagen.flow_from_directory('dataset/test_set',
                                                target_size=(64,64),
                                                batch_size=32,
                                                class_mode='binary')
model.fit_generator(training_data,
                    steps_per_epoch=8000,
                    epochs=25,
                    validation_data=test_data,
                    validation_steps=2000,
                    callbacks=[checkpoint])
#Loading The best Weights
model.load_weights('weight.hdf5')
#saving our CNN model
model.save('cnn_catdog.h5')

#testing our CNN model
#this a cat image not from our dataset let's see
catimg=plt.imread('/content/images cat .jpeg')
plt.imshow(catimg)
#predicting
from tensorflow.keras.preprocessing import image
import numpy as np
img = image.load_img('/content/images cat .jpeg', target_size=(64, 64))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
print('it is a '+categories[int(np.round(model.predict(img_tensor)))])

#beng0o:) 














