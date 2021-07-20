from mts.core.mtserie_dataset import MTSerieDataset
from mts.core.projections import ProjectionAlg
from models.emotion_dataset_controller import *


appController = AppController()

appController.loadLocalDataset('emotions_in_music')
