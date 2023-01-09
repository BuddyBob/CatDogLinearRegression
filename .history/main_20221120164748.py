import os
import cv2

test_lst = []
train_lst = []

for i in os.listdir("./test_set"):
    for c in os.listdir("./cats"):
        print(c)
