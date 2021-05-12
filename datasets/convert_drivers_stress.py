import numpy as np
import pandas as pd
import scipy.io
import os
import sys
sys.path.insert(0,'..')
from mts.core.mtserie_dataset import MTSerieDataset
from mts.core.mtserie import MTSerie


def scaleToRange(OldValue, OldMin, OldMax, NewMin, NewMax):
    OldRange = (OldMax - OldMin)  
    NewRange = (NewMax - NewMin)  
    NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    return NewValue

FILE_PATH = "/home/texs/Documents/Datasets/AffectiveRoad/AffectiveROAD_Data/Database/Subj_metric/"
SAVE_DIR = "DRIVERS_STRESS"

files = [
    "SM_AD1.csv",
    "SM_BK1.csv",
    "SM_EK1.csv",
    "SM_GM1.csv",
    "SM_GM2.csv",
    "SM_KSG1.csv",
    "SM_MT1.csv",
    "SM_NM1.csv",
    "SM_NM2.csv",
    "SM_NM3.csv",
    "SM_RY1.csv",
    "SM_RY2.csv",
    "SM_SJ1.csv"
]


if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

N = len(files)

"""
minLen = np.inf
for i in range(N):
    id = files[i][:4]
    dataframe = pd.read_csv(FILE_PATH + files[i], sep = ";")
    ratings = dataframe.to_numpy().reshape([-1])
    if minLen > ratings.shape[0]:
        minLen = ratings.shape[0]
print("minLen: {}".format(minLen))
"""
dataset = MTSerieDataset()

minLen = 11103

for i in range(N):
    id = files[i][:-4]
    driveNumber = id[3:][-1]
    id = id[3:][:-1]
    id = "drive_{}_of_{}".format(driveNumber, id)
    
    dataframe = pd.read_csv(FILE_PATH + files[i], sep = ";")
    ratings = dataframe.to_numpy().reshape([-1])[:minLen]
    mtserie = MTSerie.fromDict({'Stress': ratings})
    mtserie.identifiers = {
        'id': id
    }
    dataset.add(mtserie, id)

dataset.exportToEmotionJson(
    "drivers_stress", 
    isCategorical= False, 
    minValue= 0.0,
    maxValue= 1.0,
    saveDir = SAVE_DIR,
)