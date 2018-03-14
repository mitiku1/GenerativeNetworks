from keras.layers import Conv2D,MaxPool2D,Dense,UpSampling2D,Input
from keras.models import Model,model_from_json
import json
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
    # model_json = model.to_json()
    # with open("model.json","w+") as json_file:
    #     json_file.write(model_json)
    
    # exit(0)
    return model
def load_model(json_file,weights):
    with open(json_file) as j_file:
        model_json = j_file.read()
        model = model_from_json(model_json)
        model.load_weights(weights)
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
def load_aflw_dataset(dataset_path,inputs_shape):
    dataset = load_aflw_face_dataset_image_paths(dataset_path)
    train,test = train_test_split(dataset,test_size=0.01)
    print(len(test))
    test_images = load_images_from_paths(dataset_path, test,inputs_shape)

    return train,test_images


def load_aflw_face_dataset_image_paths(dataset_dir):
    output = []
    for img_file in os.listdir(dataset_dir):
        if not img_file.endswith(".jpg"):
            print img_file
        output.append(img_file)
    return output
def load_images_from_paths(dataset_dir,imgs_files,output_shape):
    output = np.zeros((len(imgs_files),output_shape[0],output_shape[1],output_shape[2]))
    for i in range(len(imgs_files)):
        try:
            image = cv2.imread(os.path.join(dataset_dir,imgs_files[i]))
        except:
            print "Unable to read from : ",os.path.join(dataset_dir,imgs_files[i])
        if image is None:
            continue
        try:
            image = cv2.resize(image,(output_shape[0],output_shape[1]))
        except:
            continue
        try:
            if output_shape[2]==1:
                image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                image = image.reshape(output_shape[0],output_shape[1],output_shape[2])
        except:
            continue
        output[i] = image
    return output

def vae_loss(y_true, y_pred):
    """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
    # E[log P(X|z)]
    recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
    # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
    kl = 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)

    return recon + kl


def generator(dataset_dir,image_files,batch_size,input_shape):
    image_files = np.array(image_files)
    while True:
        indexes = range(len(image_files))
        np.random.shuffle(indexes)
        for i in range(0,len(indexes)-batch_size-1,batch_size):
            current_indexes = indexes[i:i+batch_size]
            current_files = image_files[current_indexes]
            current_images = load_images_from_paths(dataset_dir,current_files,input_shape)
            current_images = current_images.astype(np.float32)/255
            yield current_images,current_images
            

def train(model,dataset_dir,train_img_files,test_images,input_shape,batch_size=32,epochs=10,steps=1000):
    # model.fit(dataset[0],dataset[0],validation_data=(dataset[1],dataset[1]),epochs=20,batch_size=32)
    model.fit_generator(generator(dataset_dir,train_img_files,batch_size,input_shape),steps_per_epoch=steps,epochs=epochs,validation_data=(test_images,test_images))

    model.save_weights("model2.h5")