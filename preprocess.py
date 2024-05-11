# importing libraries
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

path = "LATEST/dataset/Split/Cataract/Train/*"
count = 1

for file in glob.glob(path):
    print(file)
    ori_img = cv2.imread(file, 1)

    #resize the image
    resize = cv2.resize(ori_img, (224,224))
    cv2.imwrite("LATEST/V2/preprocess/resize/cataract/Cataract_" + str(count) + "_resize.jpg", resize)

    #normalize the image
    #res_img = cv2.imread(resize, 0)
    norm = np.zeros((800, 800))
    normalize = cv2.normalize(resize, norm, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite("LATEST/V2/preprocess/normalize/cataract/Cataract_" + str(count) + "_normalize.jpg", normalize)

    #grayscale image
    #gray = cv2.cvtColor(normalize, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite("LATEST/V2/preprocess/grayscale/normal/Normal_" + str(count) + "_grayscale.jpg", gray)

    count += 1
