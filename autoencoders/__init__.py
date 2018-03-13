from keras.layers import Conv2D,MaxPool2D,Dense,UpSampling2D,Input
from keras.models import Model
import keras
import os 
import cv2 
import numpy as np 
import dlib
from sklearn.model_selection import train_test_split
def auto_encoder_model(input_shape):
    input_layer = Input(shape=input_shape)
    x = Conv2D(32,(3,3),strides=(1, 1),padding='same',activation="relu")(input_layer)
    x = MaxPool2D(pool_size=(2, 2),padding='same')(x)
    x = Conv2D(64,(3,3),strides=(1, 1),padding='same',activation="relu")(x)
    x = MaxPool2D(pool_size=(2, 2),padding='same')(x)
    x = Conv2D(128,(3,3),strides=(1, 1),padding='same',activation="relu")(x)
    encoded = MaxPool2D(pool_size=(2, 2),padding='same')(x)

    x = Conv2D(128,(3,3),strides=(1, 1),padding='same',activation="relu")(encoded)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64,(3,3),strides=(1, 1),padding='same',activation="relu")(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(32,(3,3),strides=(1, 1),padding='same',activation="relu")(x)
    x = UpSampling2D(size=(2, 2))(x)
    decoded = Conv2D(input_shape[2],(3,3),strides=(1, 1),padding='same',activation="sigmoid")(x)

    model = Model(inputs=input_layer,outputs=decoded)
    model.summary()
    model.compile(loss=keras.losses.mean_absolute_error,optimizer=keras.optimizers.Adam(lr=1e-4))

    return model

def load_images(images_path,output_shape):
    imgs_files = os.listdir(images_path)
    imgs_files = imgs_files[:100000]
    print "loading images"
    output = np.zeros((len(imgs_files),output_shape[0],output_shape[1],output_shape[2]))
    for i in range(len(imgs_files)):
        image = cv2.imread(os.path.join(images_path,imgs_files[i]))
        image = cv2.resize(image,(output_shape[0],output_shape[1]))
        if output_shape[2]==1:
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            image = np.expand_dims(image,2)
        output[i] = image
    print "loaded images"
    output = output.astype(np.float32)/255
    train,test = train_test_split(output,test_size=0.2)
    return train,test

def train(model,dataset):
    model.fit(dataset[0],dataset[0],validation_data=(dataset[1],dataset[1]),epochs=20,batch_size=32)
    model.save_weights("model.h5")