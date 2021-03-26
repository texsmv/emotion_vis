from local_datasets_info import *
from mts.core.projections import euclidean_distance_matrix, mds_projection, mp_distance_matrix, compute_k_distance_matrixes, compute_distance_matrix
from mts.core.utils import mtserieQueryToJsonStr, subsetSeparationRanking, fishersDiscriminantRanking, scale_layout
from mts.core.distances import DistanceType, ts_euclidean_distance
from utils.utils import mtserie_from_json
from mts.core.mtserie_dataset import MTSerieDataset
from numpy.lib.index_tricks import CClass
from mts.core.mtserie import MTSerie
import numpy as np
import sys
import json

sys.path.append("..")


INFO_MIN_VALUES = "globalEmotionMin"
INFO_MAX_VALUES = "globalEmotionMax"
INFO_SERIES_LABELS = "seriesLabels"
INFO_LEN_TIME = "temporalLen"
INFO_LEN_INSTANCE = "instanceLen"
INFO_LEN_VARIABLES = "variablesLen"
INFO_DATES = "dates"
INFO_IS_DATED = "isDated"
INFO_DOWNSAMPLE_RULES = "downsampleRules"
INFO_IDS = "ids"
INFO_CATEGORICAL_LABELS = "categoricalLabels"
INFO_NUMERICAL_LABELS = "numericalLabels"
INFO_TYPE = "type"  # * either 'dimensional' or 'categorical'
INFO_DIMENSIONS = "dimensions"  # * only if type is dimensional
SETTINGS_EMOTIONS_LABELS = "emotionLabels"
SETTINGS_LOWER_BOUNDS = "globalEmotionLowerBound"
SETTINGS_UPPER_BOUNDS = "globalEmotionUpperBound"

DATASET_PATH = "datasets/"


class AppController:
    def __init__(self):
        self.loadedDatasets = []
        self.localDatasetsIds = [
            "case", "wesad_dimensional",  "wesad_categorical"]
        self.datasets = {}
        # this info is changed according to the proccesing make on the original data
        self.datasetsInfo = {}

    def loadLocalDataset(self, datasetId):
        if datasetId in self.loadedDatasets:
            return False

        if datasetId == "wesad_dimensional":
            path_info = wesad_path_info_dimensional
            paths = wesad_paths
        if datasetId == "wesad_categorical":
            path_info = wesad_path_info_categorical
            paths = wesad_paths
        elif datasetId == "case":
            path_info = case_path_info
            paths = case_paths

        with open('datasets/' + path_info, 'r') as file:
            dataInfoJson = file.read()
        self.initializeDataset(dataInfoJson)

        for path in paths:
            with open(DATASET_PATH + path, 'r') as file:
                jsonStr = file.read()
                print(path)
                self.addMtserieFromString(datasetId, jsonStr)

        return True

    def addMtserieFromString(self, datasetId, eml):
        mtserie = mtserie_from_json(eml)
        id = mtserie.info["id"]
        self.datasets[datasetId].add(mtserie, id)
        return id

    def removeDataset(self, datasetId):
        if not datasetId in self.loadedDatasets:
            return False

        del self.datasets[datasetId]
        self.loadedDatasets.remove(datasetId)
        return True

    def initializeDataset(self, jsonStr):
        infoDict = json.loads(jsonStr)
        datasetId = infoDict["id"]
        if datasetId in self.loadedDatasets:
            return False
        self.datasets[datasetId] = MTSerieDataset()
        self.datasetsInfo[datasetId] = infoDict
        self.loadedDatasets.append(datasetId)
        return True

    def addEmlToDataset(self, datasetId, eml):
        if not datasetId in self.loadedDatasets:
            return False
        mtserie = mtserie_from_json(eml)
        assert isinstance(mtserie, MTSerie)
        id = mtserie.info["id"]
        self.datasets[datasetId].add(mtserie, id)
        return True

    def getMTSeriesInRange(self, datasetId, ids, begin, end):
        query = self.datasets[datasetId].get_mtseries_in_range(
            begin, end, ids, procesed=True)
        resultMap = {id: self.mtserieToMap(
            query[id]) for id in list(query.keys())}
        return resultMap

    def downsampleDataset(self, datasetId, rule):
        self.datasets[datasetId].downsample_data(rule)

    def mtserieToMap(self, mtserie):
        assert isinstance(mtserie, MTSerie)
        mtserieMap = {}
        temporalVariables = {}
        for varName in mtserie.labels:
            temporalVariables[varName] = list(mtserie.get_serie(varName))
        mtserieMap["temporalVariables"] = temporalVariables
        mtserieMap["index"] = [str(idx) for idx in mtserie.dataframe.index]
        mtserieMap["metadata"] = mtserie.info
        mtserieMap["categoricalFeatures"] = list(
            mtserie.categoricalFeatures.values())
        mtserieMap["numericalFeatures"] = list(
            mtserie.numericalFeatures.values())
        mtserieMap["categoricalLabels"] = list(
            mtserie.categoricalFeatures.keys())
        mtserieMap["numericalLabels"] = list(mtserie.numericalFeatures.keys())
        return mtserieMap

    def getDatasetEmotionValues(self, datasetId, field):
        # field can be either 'min' or 'max'
        emotionsInfo = self.datasetsInfo[datasetId]["vocabulary"]["emotions"]
        return {emotion: emotionsInfo[emotion][field] for emotion in emotionsInfo.keys()}
    
    def getDatasetEmotionDimensions(self, datasetId):
        emotionsInfo = self.datasetsInfo[datasetId]["vocabulary"]["emotions"]
        return {emotion: emotionsInfo[emotion][field] for emotion in emotionsInfo.keys()}

    def getDatasetEmotions(self, datasetId):
        return list(self.datasetsInfo[datasetId]["vocabulary"]["emotions"].keys())

    # * now it only gets the procesed info, which is the same as original at the begin

    def getDatasetInfo(self, datasetId):
        dataInfo = {}
        dataInfo[INFO_IDS] = self.datasets[datasetId].ids
        dataInfo[INFO_MIN_VALUES] = self.getDatasetEmotionValues(
            datasetId, 'min')
        dataInfo[INFO_MAX_VALUES] = self.getDatasetEmotionValues(
            datasetId, 'max')
        dataInfo[INFO_LEN_INSTANCE] = self.datasets[datasetId].instanceLen
        dataInfo[INFO_LEN_VARIABLES] = self.datasets[datasetId].variablesLen
        dataInfo[INFO_LEN_TIME] = self.datasets[datasetId].get_timeLen(
            procesed=True)
        dataInfo[INFO_SERIES_LABELS] = self.getDatasetEmotions(datasetId)
        dataInfo[INFO_IS_DATED] = self.datasets[datasetId].isDataDated
        dataInfo[INFO_TYPE] = self.datasetsInfo[datasetId]["type"]
        if self.datasetsInfo[datasetId]["type"] == "dimensional":
            dataInfo[INFO_DIMENSIONS] = self.getDatasetEmotionValues(
                datasetId,
                "dimension")
        if 'categoricalMetadata' in self.datasetsInfo[datasetId]["vocabulary"]:
            print("si hay")
            dataInfo[INFO_CATEGORICAL_LABELS] = self.datasetsInfo[datasetId]["vocabulary"]['categoricalMetadata']
        if 'numericalMetadata' in self.datasetsInfo[datasetId]["vocabulary"]:
            print("si hay")
            dataInfo[INFO_NUMERICAL_LABELS] = self.datasetsInfo[datasetId]["vocabulary"]['numericalMetadata']

        if self.datasets[datasetId].isDataDated:
            dataInfo[INFO_DATES] = [
                str(date) for date in self.datasets[datasetId].get_datetimes(procesed=True)]
            dataInfo[INFO_DOWNSAMPLE_RULES] = self.datasets[datasetId].allowedDownsampleRules
        return dataInfo

    def compute_D_k(self, datasetId, begin, end, variables, distanceType=DistanceType.EUCLIDEAN):
        D_k = compute_k_distance_matrixes(list(self.datasets[datasetId].get_mtseries_in_range(
            begin, end).values()), variables, distanceType)
        return D_k

    def getMdsProjection(self, datasetId, begin, end, alphas, oldCoords=None, D_k=None, distanceType=DistanceType.EUCLIDEAN):

        # self.dataset.compute_distance_matrix(variables=variables,
        #                                      alphas=alphas,
        #                                      distanceType= DistanceType.EUCLIDEAN,
        #
        _D_k = D_k

        # compute D_K if not previously calculated
        if _D_k == None:
            _D_k = self.compute_D_k(
                datasetId, begin, end, self.getDatasetEmotions(datasetId), distanceType)

        D = compute_distance_matrix(
            _D_k, alphas, self.datasets[datasetId].instanceLen)
        self.datasets[datasetId].compute_projection(D)

        coords = np.array([self.datasets[datasetId]._projections[id]
                          for id in self.datasets[datasetId].ids])
        coords = scale_layout(coords)

        if isinstance(oldCoords, np.ndarray):
            P = coords
            Q = oldCoords
            A = P.transpose().dot(Q)
            u, s, vt = np.linalg.svd(A, full_matrices=True)
            v = vt.transpose()
            ut = u.transpose()
            r = np.sign(np.linalg.det(v.dot(ut)))
            R = v.dot(np.array([[1, 0], [0, r]])).dot(ut)
            coords = R.dot(P.transpose()).transpose()

        # return coords as dict
        coordsDict = {}
        ids = self.datasets[datasetId].ids
        for i in range(len(ids)):
            id = ids[i]
            coordsDict[id] = coords[i].tolist()

        return coordsDict, _D_k

    def doClustering(self, datasetId, coords, k=4):
        clusters = self.datasets[datasetId].cluster_projections(k, coords)
        return clusters

    def getFishersDiscriminantRanking(self, datasetId, D_ks, blueCluster, redCluster):
        ids = self.datasets[datasetId].ids

        blueIndexes = [ids.index(e) for e in blueCluster]
        redIndexes = [ids.index(e) for e in redCluster]

        j_s = fishersDiscriminantRanking(D_ks, blueIndexes, redIndexes)
        variablesRanks = {}
        variables = list(D_ks.keys())
        for varName in variables:
            variablesRanks[varName] = j_s[varName]
        return variablesRanks

    def getTemporalSummary(self, datasetId):
        values = self.datasets[datasetId].values()
        values = np.transpose(values, (2, 1, 0))  # to D, T, N shape
        summary = {}
        emotions = self.getDatasetEmotions(datasetId)
        summary['min'] = {}
        for i in range(len(emotions)):
            summary['min'][emotions[i]] = values[i].min(axis=0).tolist()

        summary['max'] = {}
        for i in range(len(emotions)):
            summary['max'][emotions[i]] = values[i].max(axis=0).tolist()

        summary['mean'] = {}
        for i in range(len(emotions)):
            summary['mean'][emotions[i]] = values[i].mean(axis=0).tolist()

        return summary

    def resetDataset(self, datasetId):
        self.datasets[datasetId].resetProcesedMtseries()
