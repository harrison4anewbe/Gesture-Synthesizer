# Emotion Classification in short texts with BERT

import pandas as pd
import numpy as np
import os

import ktrain
from ktrain import text

# import data
data_train = pd.read_csv("./dataset/train_out.csv", encoding="utf-8")
data_test = pd.read_csv("./dataset/test_out.csv", encoding="utf-8")

X_train = data_train.Text.tolist()
X_test = data_test.Text.tolist()

y_train = data_train.Emotion.tolist()
y_test = data_test.Emotion.tolist()

data = data_train.append(data_test, ignore_index=True)

class_names = ["joy", "sadness", "fear", "anger", "neutral"]

print("size of training set: %s" % (len(data_train["Text"])))
print("size of validation set: %s" % (len(data_test["Text"])))
print(data.Emotion.value_counts())
# encoding the labels
encoding = {"joy": 0, "sadness": 1, "fear": 2, "anger": 3, "neutral": 4}

# Integer values for each class
y_train = [encoding[x] for x in y_train]
y_test = [encoding[x] for x in y_test]

# data preprocessing
(x_train, y_train), (x_test, y_test), preproc = text.texts_from_array(
    x_train=X_train,
    y_train=y_train,
    x_test=X_test,
    y_test=y_test,
    class_names=class_names,
    preprocess_mode="bert",
    maxlen=350,
    max_features=35000,
)

# Training and validation
model = text.text_classifier("bert", train_data=(x_train, y_train), preproc=preproc)

learner = ktrain.get_learner(
    model, train_data=(x_train, y_train), val_data=(x_test, y_test), batch_size=12
)
learner.fit_onecycle(6e-6, 5)

res = learner.validate(val_data=(x_test, y_test), class_names=class_names)
print(res)
# get predictor and test
predictor = ktrain.get_predictor(learner.model, preproc)
predictor.get_classes()

import time

message = "Though I do not know how to deal with it, I can still work on it"
message = "That sucks"

start_time = time.time()
prediction = predictor.predict(message)

print("predicted: {} ({:.2f})".format(prediction, (time.time() - start_time)))

# Saving Bert model

predictor.save("./models/bert_")
