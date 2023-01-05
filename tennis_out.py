# Фильтрация фона видео с OpenCV Временная Медианная Фильтрация
# https://waksoft.susu.ru/2019/08/29/filtracija-fona-video-s-opencv/

# Remove Moving Objects from Video in OpenCV Python using Background Subtraction. Фильтрация по среднему
# https://machinelearningknowledge.ai/remove-moving-objects-from-video-in-opencv-python-using-background-subtraction-running-average-vs-median-filtering/

import cv2
import numpy as np


def img_2_hsv(image):
    # Convert frame from BGR to HSV color format.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Obtain HUE channel
    h = hsv[:, :, 0]

    # Applying thresholding
    threshold_value = 20
    h_copy = h.copy()
    h[h_copy > threshold_value] = 0
    h[h_copy <= threshold_value] = 1
    # The value 40 select by trail and error method.

    hsv = h * 255
    return hsv


def median_frame_func(video, frame_rate=25):
    # Randomly select 25 frames
    frame_ids = video.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(
        size=frame_rate)

    # Store selected frames in an array
    frames = []
    for frame_id in frame_ids:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        success, frame = video.read()
        frames.append(frame)

    # Calculate the median along the time axis
    median_frame = np.median(frames, axis=0).astype(dtype=np.uint8)

    # Reset frame number to 0
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    return median_frame


# Open Video
cap = cv2.VideoCapture('videos/tennis.mp4')
if not cap.isOpened():
    print('Error opening video')

# medianFrame = median_frame_func(cap)
medianFrame = cv2.imread('img/tennis0.jpg')

# Display median frame
# cv2.imshow('Median Frame', medianFrame)

# Convert background to grayscale
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

# Loop over all frames
success = True
while success:

    # Read frame
    success, frame = cap.read()

    # Convert current frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Calculate absolute difference of current frame and the median frame
    diff = cv2.absdiff(gray, grayMedianFrame)
    # cv2.imshow('frame diff', diff)

    # Threshold to binarize
    thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1]
    # Размыть изображение, чтобы очистить от шумов
    blur = cv2.medianBlur(thresh, 15)

    # Расширяем области с целью объединить их в меньшее количество
    # более крупных областей
    kernel = np.ones((5, 5),
                     np.uint8)  # Ядро вокруг которого объединение будет происходить. Что такое?
    # Расширение, укрупнение
    dilate = cv2.dilate(blur, kernel, iterations=3)

    ################ HSV #####################
    # Переводим в HSV медианный кадр без человека
    hsv_median_frame = img_2_hsv(medianFrame)
    # cv2.imshow('hsv_median_frame', hsv_median_frame)

    # Переводим в HSV кадр видео
    hsv = img_2_hsv(frame)
    # Суммируем медианный кадр без человека с кадром из видео
    summa_median = cv2.bitwise_xor(hsv_median_frame, hsv)
    # Размыть изображение, чтобы очистить от шумов
    summa_median_blur = cv2.medianBlur(summa_median, 15)
    cv2.imshow('median', summa_median_blur)

    # Суммируем маску от движения и маску HSV
    summa = cv2.bitwise_or(dilate, summa_median_blur)
    cv2.imshow('Summa', summa)
    ################ HSV #####################

    # Display image
    frame = cv2.bitwise_and(frame, frame, mask=summa)
    cv2.imshow('Original', frame)
    # cv2.imshow('blur', blur)
    cv2.imshow('dilate', dilate)

    # Quit by pressing ESC
    k = cv2.waitKey(10)
    if k == 27:  # wait for ESC key to exit
        break

# Release video object
cap.release()

# Destroy all windows
cv2.destroyAllWindows()
