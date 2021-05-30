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


DATASET_PATH = "/home/texs/Documents/Datasets/AFF-WILD/Dimensional"
SAVE_DIR = "AFF-WILD-DIMENSIONAL"

emotionsLabels = ["valence", "arousal"]
dataset = MTSerieDataset()
paths = getFiles(DATASET_PATH, allowedExtensions = (".txt"))

timeSerieSize = 2000
videoLens = []
counter = 0
for path in paths:
    counter += 1
    head, tail = os.path.split(path)
    id = "video_{}".format(counter)
    # with open(path, 'r') as f:
    #     lines = f.readlines()
    # for line in lines[1:]:
    #     print(f"- {line}")
    df = pd.read_csv(path)
    valenceSerie = df["valence"].to_numpy()[:timeSerieSize]
    arousalSerie = df["arousal"].to_numpy()[:timeSerieSize]
    if len(valenceSerie) != timeSerieSize:
        continue
    # arousalSerie = np.vectorize(scaleToRange)(arousalSerie, -1, 1, 0, 1)
    # valenceSerie = np.vectorize(scaleToRange)(valenceSerie, -1, 1, 0, 1)

    mtserie = MTSerie.fromDict({"arousal":  arousalSerie, "valence":  valenceSerie,})
    mtserie.identifiers = {
        'id': id,
    }
    dataset.add(mtserie, id)

print(dataset.instanceLen)
print(dataset.timeLen)

timeLabels = [f"frame_{i}" for i in range(timeSerieSize)]
#     videoLens += [len(valence)]

# videoLens = np.array(videoLens)
# print(np.min(videoLens))
# print(np.max(videoLens))
# print(np.mean(videoLens))
# print(np.bincount(videoLens))
# print(np.argmax(np.bincount(videoLens)))

createDir(SAVE_DIR)

dataset.exportToEmotionJson(
    "AFF-WILD-DIMENSIONAL", 
    isCategorical= False,
    minValue= -1.0,
    maxValue= 1.0,
    saveDir = SAVE_DIR,
    labels=timeLabels
)

