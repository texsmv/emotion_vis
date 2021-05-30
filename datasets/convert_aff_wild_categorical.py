import os
import sys
import numpy as np
import pandas as pd
import json
from utils import getFiles, createDir
sys.path.insert(0,'..')
from mts.core.mtserie_dataset import MTSerieDataset
from mts.core.mtserie import MTSerie

# ! video with id 127 and 381 deleted due to them having emotions in different range



def scaleToRange(OldValue, OldMin, OldMax, NewMin, NewMax):
    OldRange = (OldMax - OldMin)  
    NewRange = (NewMax - NewMin)  
    NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    return NewValue


DATASET_PATH = "/home/texs/Documents/Datasets/AFF-WILD/Categorical"
SAVE_DIR = "AFF-WILD-CATEGORICAL"

emotionsLabels = ["Neutral","Anger","Disgust","Fear","Happiness","Sadness","Surprise"]
dataset = MTSerieDataset()
paths = getFiles(DATASET_PATH, allowedExtensions = (".txt"))

timeSerieSize = 1800
videoLens = []
counter = 0
for path in paths:
    counter += 1
    head, tail = os.path.split(path)
    id = "video_{}".format(counter)
    values = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:
        emotionPoint = np.eye(7)[int(line)]
        values += [emotionPoint]
        # print(f"- {line}")
    values = np.array(values).transpose()[:, :timeSerieSize]
    if values.shape[1] != timeSerieSize:
        continue

    mtserie = MTSerie.fromDArray(values, labels = emotionsLabels)
    mtserie.identifiers = {
        'id': id,
    }
    dataset.add(mtserie, id)
    # videoLens += [values.shape[1]]

timeLabels = [f"frame_{i}" for i in range(timeSerieSize)]

# videoLens = np.array(videoLens)
# print(np.min(videoLens))
# print(np.max(videoLens))
# print(np.mean(videoLens))
# print(np.bincount(videoLens))
# print(np.argmax(np.bincount(videoLens)))
createDir(SAVE_DIR)

dataset.exportToEmotionJson(
    "AFF-WILD-CATEGORICAL", 
    isCategorical= True,
    minValue= 0.0,
    maxValue= 1.0,
    saveDir = SAVE_DIR,
    labels=timeLabels
)

