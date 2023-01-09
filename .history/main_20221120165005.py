import os
import cv2

test_lst = []
train_lst = []

for i in os.listdir("./test_set")[:3]:
    for c in os.listdir(f"./test_set/{i}"):
        print(c)
