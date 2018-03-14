from  autoencoders import train,load_aflw_dataset,auto_encoder_model,load_model
import numpy as np
import os
import cv2
import scipy
import keras
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dataset_path",default="/home/mtk/datasets/img_align_celeba/img_align_celeba",type=str)
    parser.add_argument('-i', '--input_shape', nargs=3,default=[48,48,3], action='append')
    parser.add_argument("-b","--batch_size",default=32,type=int)
    parser.add_argument("-e","--epochs",default=10,type=int)
    parser.add_argument("-s","--steps",default=1000,type=int)
    parser.add_argument("-l","--lr",default=1e-4,type=float)
    return   parser.parse_args()


def main():
    
    args = get_args()
    input_shape = tuple(args.input_shape)
    # model = auto_encoder_model(input_shape)
    dataset_path =args.dataset_path
    batch_size = args.batch_size
    epochs = args.epochs
    steps = args.steps
    lr = steps.lr


    model = load_model("model.json","model.h5")
    model.compile(loss=keras.losses.binary_crossentropy,optimizer=keras.optimizers.Adam(lr=lr))
    # dataset = load_images("/home/mtk/dataset/aflw/aflw/face",(input_shape))
    training,test = load_aflw_dataset(dataset_path,(input_shape))
    # training,test = load_aflw_dataset("/home/mtk/datasets/img_align_celeba/img_align_celeba",(input_shape))
    test = test.astype(np.float32)/255
    train(model,dataset_path,training,test, input_shape,batch_size=batch_size,epochs=epochs,steps=steps)
if __name__ == "__main__":
    main()
    # input_shape = (48,48,3)
    # image_files = os.listdir("/home/mtk/dataset/aflw/aflw/face")
    # np.random.shuffle(image_files)
    # model = auto_encoder_model(input_shape)
    # model.load_weights("model.h5")
    # for i in range(10):
    #     image = cv2.imread("/home/mtk/dataset/aflw/aflw/face/"+image_files[i])
    #     image =cv2.resize(image,(48,48))
    #     if input_shape[2]==1:
    #         image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #         image = np.expand_dims(image,2)
    #     im = np.expand_dims(image,0).astype(np.float32)/255
    #     reco = model.predict(im)[0] * 255
    #     reco = reco.astype(np.uint8)
        
    #     reco = np.squeeze(reco)
        
    #     cv2.imshow("Reco",reco)
    #     cv2.imshow("Orig",image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()