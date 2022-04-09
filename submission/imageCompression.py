import numpy as np
import cv2

def imageCompressionProcess(image):
    resizedImage = cv2.resize(image, (21, 24))
    # print(resizedImage.shape)
    reshapedImage = np.reshape(resizedImage, (21, 24))
    #print(reshapedImage.shape)
    reshapedImage = reshapedImage / 255
    # if (timestep == 50):
    #     cv2.imshow("Resized Image", resizedImage)
    #     cv2.waitKey(0)
    reshapedImage = reshapedImage.flatten()
    return reshapedImage