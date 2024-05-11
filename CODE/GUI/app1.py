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
st.set_page_config(page_title="CATEDIS 2", page_icon="catedis.png")

model = load_model('./model/dataset45_30.h5', compile=False) #K7

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
def preprocess (img_path):
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
            icons=["upload"],  # optional
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
        img_file = st.file_uploader("Upload image of an eye", type=["jpg", "png", "jpeg", "jfif"])

        if 'upload' not in st.session_state:
            st.session_state.upload = " "

        if img_file is not None:
            st.image(img_file, use_column_width=False, width=300)
            st.session_state.upload = img_file.name
            with open(st.session_state.upload, "wb") as f:
                f.write(img_file.getbuffer())

            img = Image.open(st.session_state.upload)
            print(st.session_state.upload)

            # get width and height
            width = img.width
            height = img.height

            # display width and height
            st.text("Height : " + str(height))
            st.text("Width : " + str(width))

    if selected == "Preprocess":
        st.header("Pre process")

        #st.subheader("Image resized and normalized")
        pp = preprocess(st.session_state.upload)

        st.image(pp[0], caption='Uploaded image', channels='RGB', width=150)
        st.image(pp[1], caption='Resized image', channels='RGB', width=150)
        st.image(pp[2], caption='Normalized image', channels='RGB', width=150)
        cv2.imwrite("seg.jpg", pp[2])

        if 'preprocessed' not in st.session_state:
            st.session_state.preprocessed = " "

        st.session_state.preprocessed = pp[2]

    if selected == "Segmentation":
        st.header("Segmentation")
       #st.subheader("K-means clustering, k=7")
        pp = preprocess(st.session_state.upload)
        seg = segmentation("seg.jpg")
        st.image(pp[2], caption='Normalized image', width=224, use_column_width=False)
        st.image(seg[1], caption='Segmented image', width=224, use_column_width=False)

    if selected == "Classification":
        st.header("Classification")

        result = classify(st.session_state.upload)
        st.success("Accuracy : " + str(round(result[1], 2)) + " %")
        st.info("Result : Cataract " + result[0])

        st.caption("0 : Cataract detected")
        st.caption("1 : Cataract not detected")
        print(result[2])
        print(result[3])
        data = {"Prediction": [result[2], result[3]]}
        df = pd.DataFrame(data)
        st.bar_chart(data=df)
        print("######################################")

run()