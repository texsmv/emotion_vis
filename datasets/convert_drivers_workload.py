import numpy as np
import pandas as pd
import scipy.io
import os
import sys
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

FILE_PATH = "/home/texs/Documents/Datasets/DriversWorkload/hcilab_driving_dataset/dataset_web/"
SAVE_DIR = "DRIVERS_WORKLOAD"

files = ["participant_1.csv", "participant_2.csv", "participant_3.csv", "participant_4.csv",
        "participant_5.csv", "participant_6.csv", "participant_7.csv", "participant_8.csv",
        "participant_9.csv", "participant_10.csv"]

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

N = len(files)

# """
# minLen = np.inf
# for i in range(N):
#     id = files[i]
#     dataframe = pd.read_csv(FILE_PATH + files[i], sep = ";")
#     day = '10/02/21 '
#     ratings = dataframe['Rating_Videorating'].to_numpy()
#     times = dataframe['Time_Light'].to_numpy()
#     times = [datetime.strptime(day + f, '%d/%m/%y %H:%M:%S:%f')  for f in times]
#     dataframe = pd.DataFrame({'Rating_Videorating': ratings}, index=times)
#     dataframe = dataframe.resample('S').mean()
#     ratings = dataframe['Rating_Videorating'].to_numpy()
#     if minLen > ratings.shape[0]:
#         minLen = ratings.shape[0]
# print("min")
# print(minLen)
# """


dataset = MTSerieDataset()

minLen = 1565
for i in range(N):
    id = files[i][:-4]
    dataframe = pd.read_csv(FILE_PATH + files[i], sep = ";")
    ratings = dataframe['Rating_Videorating'].to_numpy()
    times = dataframe['Time_Light'].to_numpy()
    day = '10/02/21 '
    times = [datetime.strptime(day + f, '%d/%m/%y %H:%M:%S:%f')  for f in times]
    dataframe = pd.DataFrame({'Rating_Videorating': ratings}, index=times)
    dataframe = dataframe.resample('S').mean()
    ratings = dataframe['Rating_Videorating'].to_numpy()[: minLen]
    times = dataframe.index.to_numpy()[: minLen]
    ratings = np.vectorize(scaleToRange)(ratings, 0, 1000, 0, 1)
    mtserie = MTSerie.fromDict({'Workload': ratings}, index=times[:minLen])
    mtserie.identifiers = {
        'id': id
    }
    dataset.add(mtserie, id)

dataset.exportToEmotionJson(
    "drivers_workload",
    isCategorical= True, 
    minValue= 0.0,
    maxValue= 1.0,
    saveDir = SAVE_DIR,
)