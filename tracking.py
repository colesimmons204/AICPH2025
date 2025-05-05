import cv2
import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.models import load_model
import pyttsx3

center_flag = True
start_w = 0
start_h = 0
tts = pyttsx3.Engine()

my_model = load_model("my_model.keras")
video = cv2.VideoCapture(0)
if video.isOpened():
    while True:
        ret, frame = video.read()
        if ret:
            if(center_flag):
                h, w = frame.shape[:2]
                min_dim = min(h, w)
                min_dim -= min_dim%28
                #Thou shalt do integer division
                start_w = w//2 - min_dim//2
                start_h = h//2 - min_dim//2
                center_flag = False
            cam_in = frame[start_h:(start_h + min_dim), start_w:(start_w + min_dim)]
            cam_in =cv2.resize(cam_in, (28, 28))
            cv2.imshow('frame', cam_in)
            cam_in = cam_in / 255.0
            pred = my_model(cam_in.reshape(-1,28,28,1))
            print(pred)
            #Returns the best prediction as its capital letter
            letter = chr(ord("@") + np.argmax(pred[0]))
            confidence = pred[0, np.argmax(pred[0])]
            if confidence > .9:
                tts.say(letter)
                tts.runAndWait()
            a = cv2.waitKey(10)
            #escape key quits the program
            if a == 27:
                print("oops")
                break
        else:
            print("Cannot open camera") 
            break
    video.release()
