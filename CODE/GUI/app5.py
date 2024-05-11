import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
from tensorflow.keras.utils import load_img,img_to_array
from keras.models import load_model
import numpy as np
import scipy.ndimage as ndimage
import cv2
import matplotlib.pyplot as plt
from skimage.transform import hough_circle, hough_circle_peaks, hough_ellipse
from skimage.feature import canny
from skimage.draw import circle_perimeter
import glob
import math
import pandas as pd
st.set_page_config(page_title="CATEDIS", page_icon="catedis.png")

model = load_model('./model/dataset45_30.h5', compile=False) #K9

lab = {0: 'cataract', 1: 'no cataract'}

def classify(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    #img = img.reshape((224, 224, 1))
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = lab[y]
    print (answer)
    print (y)
    c = answer[0, 0]
    nc = answer[0, 1]

    if y==1 :
        acc = 100 * (answer[0, 1] / 1)
        cat = "not detected"
        print(acc)
        print(answer[0,1])
    else:
        acc = 100 * (answer[0, 0] / 1)
        cat = "detected"
        print(answer[0, 0])
        print(acc)

    return cat, acc, c, nc

#image preprocessing
def prep (img_path):
    # resize the image
    ori = cv2.imread(img_path)
    receive = cv2.cvtColor(ori, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(ori, cv2.COLOR_BGR2RGB)
    resize = cv2.resize(image, (224, 224))

    #normalize the image
    norm = np.zeros((800, 800))
    normalize = cv2.normalize(resize, norm, 0, 255, cv2.NORM_MINMAX)

    return receive, resize, normalize

#image segmentation
def segmentation (img_path):
    image = cv2.imread(img_path)

    # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
    pixel_vals = image.reshape((-1, 3))

    # Convert to float type
    pixel_vals = np.float32(pixel_vals)

    # the below line of code defines the criteria for the algorithm to stop running,
    # which will happen is 100 iterations are run or the epsilon (which is the required accuracy)
    # becomes 85%
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

    # then perform k-means clustering wit h number of clusters defined as 3
    # also random centres are initially choosed for k-means clustering
    k = 5

    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # convert data into 8-bit values
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]

    # reshape data into the original image dimensions

    segmented_image = segmented_data.reshape((image.shape))
    return image, segmented_image


def run():

    with st.sidebar:
        selected = option_menu(
            menu_title="CATEDIS",  # required
        options=["Upload image", "Preprocess", "Segmentation", "Classification"],  # required
            menu_icon="cast",  # optional
            orientation="vertical",
            default_index=0,  # optional

            styles={
                "container": {"padding": "0!important", "background-color": "dark blue"},
                "icon": {"color": "blue", "font-size": "18px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "gray",
                },
                "nav-link-selected": {"background-color": "light gray"},
            },
        )

    if selected == "Upload image":
        save_image_path = "logo.png"
        st.title("CATARACT EYE DISEASE DETECTION USING CONVOLUTIONAL NEURAL NETWORK")
        st.header("Upload image")
        #first image
        img_file1 = st.file_uploader("Upload image 1", type=["jpg", "png", "jpeg", "jfif"], key="count1")

        if 'upload1' not in st.session_state:
            st.session_state.upload1 = " "

        if img_file1 is not None:
            st.image(img_file1, use_column_width=False, width=300)
            st.session_state.upload1 = img_file1.name
            with open(st.session_state.upload1, "wb") as f:
                f.write(img_file1.getbuffer())

            img1 = Image.open(st.session_state.upload1)
            print(st.session_state.upload1)

            # display width and height
            st.text("Height : " + str(img1.height))
            st.text("Width : " + str(img1.width))

        #second image
        img_file2 = st.file_uploader("Upload image 2", type=["jpg", "png", "jpeg", "jfif"], key="count2")

        if 'upload2' not in st.session_state:
            st.session_state.upload2 = " "

        if img_file2 is not None:
            st.image(img_file2, use_column_width=False, width=300)
            st.session_state.upload2 = img_file2.name
            with open(st.session_state.upload2, "wb") as f:
                f.write(img_file2.getbuffer())

            img2 = Image.open(st.session_state.upload2)
            print(st.session_state.upload2)

            # display width and height
            st.text("Height : " + str(img2.height))
            st.text("Width : " + str(img2.width))

        #third image
        img_file3 = st.file_uploader("Upload image 3", type=["jpg", "png", "jpeg", "jfif"], key="count3")

        if 'upload3' not in st.session_state:
            st.session_state.upload3 = " "

        if img_file3 is not None:
            st.image(img_file3, use_column_width=False, width=300)
            st.session_state.upload3 = img_file3.name
            with open(st.session_state.upload3, "wb") as f:
                f.write(img_file3.getbuffer())

            img3 = Image.open(st.session_state.upload3)
            print(st.session_state.upload3)

            # display width and height
            st.text("Height : " + str(img3.height))
            st.text("Width : " + str(img3.width))

        #fourth image
        img_file4 = st.file_uploader("Upload image 4", type=["jpg", "png", "jpeg", "jfif"], key="count4")

        if 'upload4' not in st.session_state:
            st.session_state.upload4 = " "

        if img_file4 is not None:
            st.image(img_file4, use_column_width=False, width=300)
            st.session_state.upload4 = img_file4.name
            with open(st.session_state.upload4, "wb") as f:
                f.write(img_file4.getbuffer())

            img4 = Image.open(st.session_state.upload4)
            print(st.session_state.upload4)

            # display width and height
            st.text("Height : " + str(img4.height))
            st.text("Width : " + str(img4.width))

        #fifth image
        img_file5 = st.file_uploader("Upload image 5", type=["jpg", "png", "jpeg", "jfif"], key="count5")

        if 'upload5' not in st.session_state:
            st.session_state.upload5 = " "

        if img_file5 is not None:
            st.image(img_file5, use_column_width=False, width=300)
            st.session_state.upload5 = img_file5.name
            with open(st.session_state.upload5, "wb") as f:
                f.write(img_file5.getbuffer())

            img5 = Image.open(st.session_state.upload5)
            print(st.session_state.upload5)

            # display width and height
            st.text("Height : " + str(img5.height))
            st.text("Width : " + str(img5.width))

    if selected == "Preprocess":
        print("prep")
        st.header("Pre process")
        print("prep")

        st.info("Image 1")
        pp1 = prep(st.session_state.upload1)
        st.image(pp1[0], caption='Uploaded image', channels='RGB', width=150)
        st.image(pp1[1], caption='Resized image', channels='RGB', width=150)
        st.image(pp1[2], caption='Normalized image', channels='RGB', width=150)
        cv2.imwrite("seg1.jpg", pp1[2])

        #second image
        st.info("Image 2")
        pp2 = prep(st.session_state.upload2)
        st.image(pp2[0], caption='Uploaded image', channels='RGB', width=150)
        st.image(pp2[1], caption='Resized image', channels='RGB', width=150)
        st.image(pp2[2], caption='Normalized image', channels='RGB', width=150)
        cv2.imwrite("seg2.jpg", pp2[2])

        #third image
        st.info("Image 3")
        pp3 = prep(st.session_state.upload3)
        st.image(pp3[0], caption='Uploaded image', channels='RGB', width=150)
        st.image(pp3[1], caption='Resized image', channels='RGB', width=150)
        st.image(pp3[2], caption='Normalized image', channels='RGB', width=150)
        cv2.imwrite("seg3.jpg", pp3[2])

        #fourth image
        st.info("Image 4")
        pp4 = prep(st.session_state.upload4)
        st.image(pp4[0], caption='Uploaded image', channels='RGB', width=150)
        st.image(pp4[1], caption='Resized image', channels='RGB', width=150)
        st.image(pp4[2], caption='Normalized image', channels='RGB', width=150)
        cv2.imwrite("seg4.jpg", pp4[2])

        #fifth image
        st.info("Image 5")
        pp5 = prep(st.session_state.upload5)
        st.image(pp5[0], caption='Uploaded image', channels='RGB', width=150)
        st.image(pp5[1], caption='Resized image', channels='RGB', width=150)
        st.image(pp5[2], caption='Normalized image', channels='RGB', width=150)
        cv2.imwrite("seg5.jpg", pp5[2])

    if selected == "Segmentation":
        st.header("Segmentation")
        #st.subheader("K-means clustering, k=5")

        # first image
        st.info("Image 1")
        pp1 = prep(st.session_state.upload1)
        seg1 = segmentation("seg1.jpg")
        st.image(pp1[2], caption='Normalized image', width=224, use_column_width=False)
        st.image(seg1[1], caption='Segmented image', width=224, use_column_width=False)

        #second image
        st.info("Image 2")
        pp2 = prep(st.session_state.upload2)
        seg2 = segmentation("seg2.jpg")
        st.image(pp2[2], caption='Normalized image', width=224, use_column_width=False)
        st.image(seg2[1], caption='Segmented image', width=224, use_column_width=False)

        #third image
        st.info("Image 3")
        pp3 = prep(st.session_state.upload3)
        seg3 = segmentation("seg3.jpg")
        st.image(pp3[2], caption='Normalized image', width=224, use_column_width=False)
        st.image(seg3[1], caption='Segmented image', width=224, use_column_width=False)

        #fourth image
        st.info("Image 4")
        pp4 = prep(st.session_state.upload4)
        seg4 = segmentation("seg4.jpg")
        st.image(pp4[2], caption='Normalized image', width=224, use_column_width=False)
        st.image(seg4[1], caption='Segmented image', width=224, use_column_width=False)

        #fifth image
        st.info("Image 5")
        pp5 = prep(st.session_state.upload5)
        seg5 = segmentation("seg5.jpg")
        st.image(pp5[2], caption='Normalized image', width=224, use_column_width=False)
        st.image(seg5[1], caption='Segmented image', width=224, use_column_width=False)

    if selected == "Classification":
        st.header("Classification")

        result1 = classify(st.session_state.upload1)

        print("image 1")
        print(result1[2])
        print(result1[3])

        result2 = classify(st.session_state.upload2)

        print("image 2")
        print(result2[2])
        print(result2[3])

        result3 = classify(st.session_state.upload3)

        print("image 3")
        print(result3[2])
        print(result3[3])

        result4 = classify(st.session_state.upload4)

        print("image 4")
        print(result4[2])
        print(result4[3])

        result5 = classify(st.session_state.upload5)

        print("image 5")
        print(result5[2])
        print(result5[3])

        average = (result1[1] + result2[1] + result3[1] + result4[1] + result5[1]) / 5

        if result1[0] == result2[0] == result3[0] == result4[0] == result5[0]:
           st.success(" The accuracy in average : " + str(round(average, 2)) + " %")
           st.info("Therefore, cataract " + str(result1[0]))
        else:
            st.error("This is error messages")
            st.error("Image 1 : Cataract " + str(result1[0]) + " " + str(result1[1]))
            st.error("Image 2 : Cataract " + str(result2[0]) + " " + str(result2[1]))
            st.error("Image 3 : Cataract " + str(result3[0]) + " " + str(result3[1]))
            st.error("Image 4 : Cataract " + str(result4[0]) + " " + str(result4[1]))
            st.error("Image 5 : Cataract " + str(result5[0]) + " " + str(result5[1]))

        print("ALL IMAGES CLASSIFIED")
        print("$$$$$$$$$$$$$$$$$$$$$$$$$")

run()