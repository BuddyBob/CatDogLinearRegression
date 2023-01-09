import os
import cv2
import matplotlib.pyplot as plt

test_lst = []
train_lst = []

for i in os.listdir("./test_set")[:2]:
    for c in os.listdir(f"./test_set/{i}"):
        img = cv2.imread(f"./test_set/{i}/{c}", cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        test_lst.append(img)
        
for i in os.listdir("./training_set")[:2]:
    for c in os.listdir(f"./training_set/{i}"):
        img = cv2.imread(f"./training_set/{i}/{c}", cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        train_lst.append(img)

plt.figure(figsize=(10, 10))
count = 0
for animal in test_lst[:9]:
    plt.subplot(3,3, count+1)
    plt.imshow(animal)
    count += 1
plt.show()