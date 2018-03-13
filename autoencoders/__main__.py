from  autoencoders import train,load_images,auto_encoder_model
import numpy as np
import os
import cv2
import scipy


def main():
    input_shape = (48,48,1)
    model = auto_encoder_model(input_shape)
    dataset = load_images("/home/mtk/dataset/aflw/aflw/face",(input_shape))
    dataset = dataset
    train(model,dataset)
if __name__ == "__main__":
    main()
    # input_shape = (48,48,1)
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