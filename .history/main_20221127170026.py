import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib
x_test = []
y_test = []
x_train = []
y_train = []

#dog = 0
#cat = 1
for i in os.listdir("./test_set")[:2]:
    for c in os.listdir(f"./test_set/{i}"):
        img = cv2.imread(f"./test_set/{i}/{c}")
        img = cv2.resize(img, (64, 64))
        x_test.append(img.flatten())
        
        if i == "dogs":
            y_test.append(0)
        else:
            y_test.append(1)
        
for i in os.listdir("./training_set")[:2]:
    for c in os.listdir(f"./training_set/{i}"):
        img = cv2.imread(f"./training_set/{i}/{c}")
        img = cv2.resize(img, (64, 64))
        x_train.append(img.flatten())
        
        if i == "dogs":
            y_train.append(0)
        else:
            y_train.append(1)


new_train = np.array(x_train)/255
model = LogisticRegression()
y_train = np.array(y_train)


model.fit(new_train, y_train)
joblib.dump(model, "catdog.pkl")

# model = joblib.load("catdog.pkl")








y_pred = model.predict(np.array(x_test)/255)

accuracy = metrics.accuracy_score(y_test, y_pred)
matrix = metrics.confusion_matrix(y_test, y_pred)
print(accuracy)
print(matrix)
#(8005, 128, 128, 3)