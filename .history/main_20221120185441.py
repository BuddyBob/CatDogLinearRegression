import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
x_test = []
y_test = []
x_train = []
y_train = []

#dog = 0
#cat = 1
for i in os.listdir("./test_set")[:2]:
    for c in os.listdir(f"./test_set/{i}"):
        img = cv2.imread(f"./test_set/{i}/{c}")
        img = cv2.resize(img, (128, 128))
        x_test.append(img)
        
        if i == "dogs":
            y_test.append(0)
        else:
            y_test.append(1)
        
for i in os.listdir("./training_set")[:2]:
    for c in os.listdir(f"./training_set/{i}"):
        img = cv2.imread(f"./training_set/{i}/{c}")
        img = cv2.resize(img, (128, 128))
        x_train.append(img)
        
        if i == "dog":
            y_train.append(0)
        else:
            y_train.append(1)

# plt.figure(figsize=(10, 10))
# count = 0
# for animal in test_lst[:9]:
#     plt.subplot(3,3, count+1)
#     plt.imshow(animal)
#     count += 1
# plt.show()


new_train = np.array(x_train)/255

model = LogisticRegression(max_iter=400)

y_train = np.array(y_train)

nsamples, nx, ny, nz = new_train.shape
d2_train_dataset = new_train.reshape((nsamples,nx*ny*nz))

print(d2_train_dataset.shape, y_train.shape)
model.fit(d2_train_dataset, y_train)

# y_pred = model.predict(x_test)

# accuracy = metrics.accuracy_score(y_test, y_pred)