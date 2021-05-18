import numpy as np
import pandas as pd
import scipy.io
import os
import sys
import re

sys.path.insert(0,'..')
from mts.core.mtserie_dataset import MTSerieDataset
from mts.core.mtserie import MTSerie
from datetime import datetime, timedelta
from datetime import date


def scaleToRange(OldValue, OldMin, OldMax, NewMin, NewMax):
    OldRange = (OldMax - OldMin)  
    NewRange = (NewMax - NewMin)  
    NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    return NewValue

"""
    https://cvml.unige.ch/databases/emoMusic/

    values in range [-1, 1]
"""

FILE_PATH = "/home/texs/Documents/Datasets/EmotionInMusic/"
SAVE_DIR = "EMOTION_IN_MUSIC"

infoDf = pd.read_csv(FILE_PATH + "songs_info.csv", sep = ",")
arousalDf = pd.read_csv(FILE_PATH + "arousal_cont_average.csv", sep = ",")
valenceDf = pd.read_csv(FILE_PATH + "valence_cont_average.csv", sep = ",")

variables = ["Arousal", "Valence"]

arousal = arousalDf.to_numpy()
# delete ids
arousal = np.delete(arousal, 0, axis=1)

valence = valenceDf.to_numpy()
# delete ids
valence = np.delete(valence, 0, axis=1)

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

N = valence.shape[0]
T = valence.shape[1]
dataset = MTSerieDataset()
categoricalMetadata = {
    'genre': [re.sub(r"[\t]*", "", e) for e in infoDf["Genre"].unique().tolist()]
}
labels = ["{}s".format(i * 0.5) for i in range(T)]

for i in range(N):
    id = "song_{}".format(infoDf["file_name"].to_numpy()[i][1:-4])
    print(id)
    arousalSerie = arousal[i]
    arousalSerie = np.vectorize(scaleToRange)(arousalSerie, -1, 1, 0, 1)
    valenceSerie = valence[i]
    valenceSerie = np.vectorize(scaleToRange)(valenceSerie, -1, 1, 0, 1)
    mtserie = MTSerie.fromDict(
        {variables[0]: arousalSerie, variables[1]: valenceSerie},
        categoricalFeatures= {
            'genre': re.sub(r"[\t]*", "", infoDf["Genre"].to_numpy()[i])
        }
    )
    mtserie.identifiers = {
        'id': id,
        'artist' : re.sub(r"[\t]*", "", infoDf["Artist"].to_numpy()[i]),
        'song title' : re.sub(r"[\t]*", "", infoDf["Song title"].to_numpy()[i])
    }
    dataset.add(mtserie, id,)

dataset.exportToEmotionJson(
    "emotions_in_music", 
    isCategorical= False, 
    minValue= 0.0,
    maxValue= 1.0,
    saveDir = SAVE_DIR,
    labels=labels,
    categoricalMetadata= categoricalMetadata
)
