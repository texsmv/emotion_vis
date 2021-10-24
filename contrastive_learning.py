from models.emotion_dataset_controller import *
from contrastive import CPCA
import numpy as np


appController = AppController()
datasetId = "emotions_in_music"

appController.loadLocalDataset(datasetId)
dataset = appController.datasets[datasetId]
appController.getProjection(datasetId, 0, dataset.timeLen)
representations = appController.mts_representations[datasetId]

assert isinstance(dataset, MTSerieDataset)

values = dataset.values()

category = dataset.categoricalLabels[0]


print(values.shape)
print(representations.shape)

all_labels = np.array([mts.categoricalFeatures[category] for mts in dataset.get_mtseries()])
labels = np.unique(all_labels)
print(labels)

genreA = labels[-1]
groupA = []
groupB = []

for i in range(len(all_labels)):
    if all_labels[i] == genreA:
        groupA += [representations[i]]
    else:
        groupB += [representations[i]]

groupA = np.array(groupA)
groupB = np.array(groupB)

print(groupA.shape)
print(groupB.shape)

mdl = CPCA()
projected_data = mdl.fit_transform(groupB, groupA, gui=True)
print(projected_data)