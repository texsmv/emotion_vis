import os
import sys
import numpy as np
import json
from utils import getFiles
from utils import createDir
sys.path.insert(0,'..')
from mts.core.mtserie_dataset import MTSerieDataset
from mts.core.mtserie import MTSerie

def scaleToRange(OldValue, OldMin, OldMax, NewMin, NewMax):
    OldRange = (OldMax - OldMin)  
    NewRange = (NewMax - NewMin)  
    NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    return NewValue


DATASET_PATH = "/home/texs/Documents/Datasets/AFEW-VA"
SAVE_DIR = "AFEW-VA"

dataset = MTSerieDataset()
paths = getFiles(DATASET_PATH, allowedExtensions = (".json"))

timeSerieSize = 34

videoLens = []
for path in paths:
    head, tail = os.path.split(path)
    id = "film_{}".format(tail[:-5])
    with open(path, 'r') as f:
        jsonData = json.load(f)
        
    arousalSerie = []
    valenceSerie = []
    names = []
    
    frameNames = jsonData["frames"].keys()
    frames = jsonData["frames"]
    if len(frameNames) < timeSerieSize:
        continue

    for name in frameNames:
        arousalSerie += [frames[name]["arousal"]]
        valenceSerie += [frames[name]["valence"]]
        names += [name]
    
    arousalSerie = np.array(arousalSerie)[:timeSerieSize]
    valenceSerie = np.array(valenceSerie)[:timeSerieSize]
    names = names[:timeSerieSize]
    arousalSerie = np.vectorize(scaleToRange)(arousalSerie, -10, 10, 0, 1)
    valenceSerie = np.vectorize(scaleToRange)(valenceSerie, -10, 10, 0, 1)

    mtserie = MTSerie.fromDict({"arousal":  arousalSerie, "valence":  valenceSerie,})
    mtserie.identifiers = {
        'id': id,
        'actor' : jsonData["actor"],
    }
    dataset.add(mtserie, id)
    # videoLens += [len(frameNames)]


# videoLens = np.array(videoLens)
# print(np.min(videoLens))
# print(np.max(videoLens))
# print(np.mean(videoLens))
# print(np.bincount(videoLens))
# print(np.argmax(np.bincount(videoLens)))


createDir(SAVE_DIR)

dataset.exportToEmotionJson(
    "AFEW_VA", 
    isCategorical= False,
    minValue= 0.0,
    maxValue= 1.0,
    saveDir = SAVE_DIR,
    labels=names
)

