import numpy as np
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

FILE_PATH = "/home/texs/Documents/Datasets/ASCERTAIN/multimodal-autoencoder-ascertain-/ASCERTAIN_Features/"
SAVE_DIR = "ASCERTAIN"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

selfReports = scipy.io.loadmat(FILE_PATH + "Dt_SelfReports.mat")
moviesOrder = scipy.io.loadmat(FILE_PATH + "Dt_Order_Movie.mat")
personalityMat = scipy.io.loadmat(FILE_PATH + "Dt_Personality.mat")

personality = personalityMat["Personality"]
permutationLists = moviesOrder["PermutationList"]
ratings = selfReports['Ratings']
ratings = np.nan_to_num(ratings)

ratings[0] = np.vectorize(scaleToRange)(ratings[0], 0, 6, 0, 1)
ratings[1] = np.vectorize(scaleToRange)(ratings[1], -3, 3, 0, 1)
ratings[2] = np.vectorize(scaleToRange)(ratings[2], 0, 6, 0, 1)
ratings[3] = np.vectorize(scaleToRange)(ratings[3], 0, 6, 0, 1)
ratings[4] = np.vectorize(scaleToRange)(ratings[4], 0, 6, 0, 1)

ratings = np.transpose(ratings, (1,0,2))

print(ratings.shape)

N, D, T = ratings.shape
labels = ['Arousal', 'Valence', 'Engagement', 'Liking', 'Familiarity']
personalityLabels = ["Extroversion", "Agreeableness", "Conscientiousness", "Emotional Stability", "Openness"]

# personality dimensions number
P = personality.shape[1]


for i in range(N):
    permutationIndex = permutationLists[i].argsort()
    for j in range(D):
        ratings[i][j] = ratings[i][j][permutationIndex]

dataset = MTSerieDataset()

for i in range(N):
    id = "student_{}".format(i + 1)
    numericalFeatures = {personalityLabels[j]: personality[i][j] for j in range(P)}
    mtserie = MTSerie.fromDArray(ratings[i], labels = labels, 
        numericalFeatures = numericalFeatures
    )
    mtserie.identifiers = {
        'id': id
    }
    dataset.add(mtserie, id)

dataset.exportToEmotionJson(
    "ascertain", 
    isCategorical= True, 
    minValue= 0.0, 
    maxValue= 1.0,
    saveDir = SAVE_DIR,
    # labels = labels,
    numericalMetadata = personalityLabels
)


