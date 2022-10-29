# image_browser.py

import glob
import PySimpleGUI as sg
from PIL import Image, ImageTk
import matplotlib as plt
from skimage.filters import threshold_otsu
import numpy as np
import cv2
import os



def parse_folder(path):
    images = glob.glob(f'{path}/*.jpg')
    return images


def load_image(path, window):
    try:
        image = Image.open(path)
        image.thumbnail((600, 600))
        photo_img = ImageTk.PhotoImage(image)
        window["image"].update(data=photo_img)
    except:
        print(f"Unable to open {path}!")


def segmentWithKMeans(imagePath):
    img = cv2.imread(imagePath)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    twoDimage = img.reshape((-1, 3))
    twoDimage = np.float32(twoDimage)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 5
    attempts = 10

    ret, label, center = cv2.kmeans(twoDimage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))

    path = "segmentedImages"
    cv2.imwrite(os.path.join(path,imagePath.split("/")[len(imagePath.split("/"))-1][:-4] + "segmentationKMeans.png"), result_image)


def segmentWithContour(imagePath):
    img = cv2.imread(imagePath)
    img = cv2.resize(img, (256, 256))

    # Preprocessing the Image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
    edges = cv2.dilate(cv2.Canny(thresh, 0, 255), None)

    # Detecting and Drawing Contours

    cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
    mask = np.zeros((256, 256), np.uint8)
    masked = cv2.drawContours(mask, [cnt], -1, 255, -1)

    #  Segmenting the Regions

    dst = cv2.bitwise_and(img, img, mask=mask)
    segmented = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

    path = "segmentedImages"
    cv2.imwrite(os.path.join(path, imagePath.split("/")[len(imagePath.split("/"))-1][:-4] + "segmentationContourDetection.png"), segmented)


def segmentWithThresholding(imagePath):
    img = cv2.imread(imagePath)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    thresh = threshold_otsu(img_gray)
    img_otsu = img_gray < thresh
    filtered = filter_image(img, img_otsu)
    path = "segmentedImages"
    cv2.imwrite(os.path.join(path, imagePath.split("/")[len(imagePath.split("/"))-1][:-4] + "segmentationThresholding.png"), filtered)


def segmentWithColorMasking(imagePath):
    img = cv2.imread(imagePath)
    # Preprocessing the Image
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

    # Define the Color Range to be Detected
    light_blue = (90, 70, 50)
    dark_blue = (128, 255, 255)
    # You can use the following values for green
    # light_green = (40, 40, 40)
    # dark_greek = (70, 255, 255)
    mask = cv2.inRange(hsv_img, light_blue, dark_blue)

    #Apply the mask
    result = cv2.bitwise_and(img, img, mask=mask)

    path = "segmentedImages"
    print()
    cv2.imwrite(os.path.join(path, imagePath.split("/")[len(imagePath.split("/"))-1][:-4] + "segmentationColorMasking.png"), result)


def filter_image(image, mask):
    r = image[:, :, 0] * mask
    g = image[:, :, 1] * mask
    b = image[:, :, 2] * mask
    return np.dstack([r, g, b])


def main():
    elements = [
        [sg.Image(key="image")],
        [
            sg.Text("Image File"),
            sg.Input(size=(30, 2), enable_events=True, key="file"),
            sg.FolderBrowse(),
        ],
        [
            sg.Button("Prev"),
            sg.Button("Next"),
            sg.Button("KMeans"),
            sg.Button("Contour"),
            sg.Button("Thresholding"),
            sg.Button("Color Masking")
        ]
    ]

    window = sg.Window("Image Viewer", elements, size=(700, 700))
    images = []
    location = 0

    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "file":
            images = parse_folder(values["file"])
            if images:
                load_image(images[0], window)
        if event == "KMeans":
            segmentWithKMeans(imagePath=images[location])
        if event == "Contour":
            segmentWithContour(imagePath=images[location])
        if event == "Thresholding":
            segmentWithThresholding(imagePath=images[location])
        if event == "Color Masking":
            segmentWithColorMasking(imagePath=images[location])
        if event == "Next" and images:
            if location == len(images) - 1:
                location = 0
            else:
                location += 1
            load_image(images[location], window)
        if event == "Prev" and images:
            if location == 0:
                location = len(images) - 1
            else:
                location -= 1
            load_image(images[location], window)

    window.close()


if __name__ == "__main__":
    main()
