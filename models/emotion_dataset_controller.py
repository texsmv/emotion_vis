from numpy.lib.index_tricks import CClass
from mts.core.mtserie import MTSerie
import numpy as np
import sys
import json

sys.path.append("..")

from mts.core.mtserie_dataset import MTSerieDataset
from utils.utils import mtserie_from_json
from mts.core.distances import DistanceType, ts_euclidean_distance
from mts.core.utils import mtserieQueryToJsonStr, subsetSeparationRanking, fishersDiscriminantRanking
from mts.core.projections import euclidean_distance_matrix, mds_projection, mp_distance_matrix, compute_k_distance_matrixes, compute_distance_matrix
from local_datasets_info import *

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

SETTINGS_ALPHAS = "alphas"
SETTINGS_EMOTIONS_LABELS = "emotionLabels"
SETTINGS_LOWER_BOUNDS = "globalEmotionLowerBound"
SETTINGS_UPPER_BOUNDS = "globalEmotionUpperBound"

DATASET_PATH = "datasets/"

class AppController:
    def __init__(self):
        self.loadedDatasets = []
        self.localDatasetsIds = ["case", "wesad"]
        self.datasets = {}
    
    def loadLocalDataset(self, datasetId):
        if datasetId in self.loadedDatasets:
            return False
        
        if datasetId == "wesad":
            paths = wesad_paths
        elif datasetId == "case":
            paths = case_paths
            
        self.datasets[datasetId] = MTSerieDataset()

        for path in paths:
            with open(DATASET_PATH + path, 'r') as file:
                jsonStr = file.read()
                self.addMtserieFromString(datasetId, jsonStr)
                
        self.loadedDatasets.append(datasetId)
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

    def initializeDataset(self, datasetId):
        # todo : add this
        # if datasetId in self.loadedDatasets:
        #     return False
        self.datasets[datasetId] = MTSerieDataset()
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
    
    def getMTSeriesInRange(self, datasetId, ids, begin, end, procesed = True):
        query = self.datasets[datasetId].get_mtseries_in_range(begin, end, ids, procesed)
        resultMap = {id: self.mtserieToMap(query[id]) for id in list(query.keys())}
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
        mtserieMap["categoricalFeatures"] = list(mtserie.categoricalFeatures.values())
        mtserieMap["numericalFeatures"] = list(mtserie.numericalFeatures.values())
        mtserieMap["categoricalLabels"] = list(mtserie.categoricalFeatures.keys())
        mtserieMap["numericalLabels"] = list(mtserie.numericalFeatures.keys())
        return mtserieMap
    
    def getDatasetInfo(self, datasetId, procesed = True):
        dataInfo = {}
        dataInfo[INFO_IDS] = self.datasets[datasetId].ids
        dataInfo[INFO_MIN_VALUES] = self.datasets[datasetId].minTemporalValues
        dataInfo[INFO_MAX_VALUES] = self.datasets[datasetId].maxTemporalValues
        dataInfo[INFO_LEN_INSTANCE] = self.datasets[datasetId].instanceLen
        dataInfo[INFO_LEN_VARIABLES] = self.datasets[datasetId].variablesLen
        dataInfo[INFO_LEN_TIME] = self.datasets[datasetId].get_timeLen(procesed)
        dataInfo[INFO_SERIES_LABELS] = self.datasets[datasetId].temporalVariables
        dataInfo[INFO_IS_DATED] = self.datasets[datasetId].isDataDated
        if self.datasets[datasetId].isDataDated:
            print(procesed)
            dataInfo[INFO_DATES] = [str(date) for date in self.datasets[datasetId].get_datetimes(procesed)]
            dataInfo[INFO_DOWNSAMPLE_RULES] = self.datasets[datasetId].allowedDownsampleRules
        return dataInfo
    
    def compute_D_k(self, datasetId, variables, procesed = True, distanceType = DistanceType.EUCLIDEAN):
        D_k = compute_k_distance_matrixes(self.datasets[datasetId].get_mtseries(procesed= procesed), variables, distanceType)
        return D_k
    
    def getMdsProjection(self, datasetId, D_k, alphas, oldCoors = None):
        
        # self.dataset.compute_distance_matrix(variables=variables,
        #                                      alphas=alphas,
        #                                      distanceType= DistanceType.EUCLIDEAN,
        #                                      )
        
        # D_k = compute_k_distance_matrixes(self.datasets[datasetId].get_mtseries(procesed= procesed), variables, distanceType)
        D = compute_distance_matrix(D_k, alphas, self.datasets[datasetId].instanceLen)
        self.datasets[datasetId].compute_projection(D)
        
        coords = np.array([self.datasets[datasetId]._projections[id] for id in self.datasets[datasetId].ids])
        
        if isinstance(self.datasets[datasetId].oldCoords, np.ndarray): 
            P = coords
            Q = self.oldCoords
            A = P.transpose().dot(Q)
            u, s, vt = np.linalg.svd(A, full_matrices=True)
            v = vt.transpose()
            ut = u.transpose()
            r = np.sign(np.linalg.det(v.dot(ut)))
            R = v.dot(np.array([[1, 0], [0, r]])).dot(ut)
            coords = R.dot(P.transpose()).transpose()
        
        coordsDict = {}
        ids = self.datasets[datasetId].ids
        for i in range(len(ids)):
            id = ids[i]
            coordsDict[id] = coords[i].tolist()
        
        self.oldCoords = coords
        
        return coordsDict
    
    def doClustering(self, datasetId, coords, k = 4):
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
    
    
class EmotionDatasetController:
    def __init__(self):
        self.dataset =  MTSerieDataset()
        self.dataInfo = {}
        self.dataSettings = {}
        
        #! deprecated
        self.minValue = None
        self.maxValue = None
        self.oldCoords = None
        self.D_k = None
        super().__init__()
        
    def downsampleData(self, rule):
        self.dataset.downsample_data(rule)
        
    def computeDataInfo(self):
        self.dataInfo[INFO_MIN_VALUES] = self.dataset.minTemporalValues
        self.dataInfo[INFO_MAX_VALUES] = self.dataset.maxTemporalValues
        self.dataInfo[INFO_LEN_INSTANCE] = self.dataset.instanceLen
        self.dataInfo[INFO_LEN_VARIABLES] = self.dataset.variablesLen
        self.dataInfo[INFO_LEN_TIME] = self.dataset.timeLen
        self.dataInfo[INFO_SERIES_LABELS] = self.dataset.temporalVariables
        self.dataInfo[INFO_IS_DATED] = self.dataset.isDataDated
        if self.dataset.isDataDated:
            self.dataInfo[INFO_DATES] = [str(date) for date in self.dataset.datetimes]
            self.dataInfo[INFO_DOWNSAMPLE_RULES] = self.dataset.allowedDownsampleRules

    def setSettings(self, settings):
        self.dataSettings = settings
        print(self.dataSettings)

    def addEml(self, eml):
        mtserie = mtserie_from_json(eml)
        assert isinstance(mtserie, MTSerie)
        id = mtserie.info["id"]
        self.dataset.add(mtserie, id)
        return id
        
    def getIds(self):
        return self.dataset.ids()
    
    def removeVariables(self, names):
        for name in names:
            self.dataset.removeVariable(name)
        
    
    def calculateValuesBounds(self):
        X = self.dataset.getValues()
        assert isinstance(X, np.ndarray)
        self.minValue = X.min()
        self.maxValue = X.max()
        
    def getValuesBounds(self):
        if self.minValue != None and self.maxValue != None:
            return [self.minValue, self.maxValue]
        return [-1 ,-1]
    
    def setValuesBounds(self, minVal, maxVal):
        self.minValue = minVal
        self.maxValue = maxVal
    
    def getAllValuesInRange(self, begin, end):
        return self.dataset.queryAllByIndex(beginIndex=begin, endIndex=end, toList=True)
    
    def getTimeLength(self):
        return self.dataset.getTimeLength()
    
    def getInstanceLength(self):
        return self.dataset.getInstanceLength()
    
    def getVariablesNames(self):
        return self.dataset.getVariablesNames()
    
    def getNumericalLabels(self):
        return self.dataset.getNumericalLabels()

    def getCategoricalLabels(self):
        return self.dataset.getCategoricalLabels()
    
    def queryAllInRange(self, begin, end):
        indexBegin = self.dataset.first.dataframe.index[begin]
        indexEnd = self.dataset.first.dataframe.index[end]
        result = self.dataset.query_all_by_range(indexBegin, indexEnd)
        print(result)
        resultMap = {id: self.mtserieToMap(result[id]) for id in list(result.keys())}
        print(resultMap)
        #  mtserieQueryToJsonStr(self.dataset.queryAllByIndex(begin, end, toList=True))
        return resultMap
    
    def mtserieToMap(self, mtserie):
        assert isinstance(mtserie, MTSerie)
        mtserieMap = {}
        temporalVariables = {}
        for varName in mtserie.labels:
            temporalVariables[varName] = list(mtserie.get_serie(varName))
        mtserieMap["temporalVariables"] = temporalVariables
        mtserieMap["index"] = [str(idx) for idx in mtserie.dataframe.index]
        mtserieMap["metadata"] = mtserie.info
        mtserieMap["categoricalFeatures"] = list(mtserie.categoricalFeatures.values())
        mtserieMap["numericalFeatures"] = list(mtserie.numericalFeatures.values())
        mtserieMap["categoricalLabels"] = list(mtserie.categoricalFeatures.keys())
        mtserieMap["numericalLabels"] = list(mtserie.numericalFeatures.keys())
        return mtserieMap
        
    def getSubsetsDimensionsRankings(self, blueCluster, redCluster):
        ids = self.dataset.ids
        
        blueIndexes = [ids.index(e) for e in blueCluster]
        redIndexes = [ids.index(e) for e in redCluster]
        
        j_s = subsetSeparationRanking(self.D_k, blueIndexes, redIndexes)
        variablesRanks = {}
        vnames = self.dataSettings[SETTINGS_EMOTIONS_LABELS]
        for i in range(len(vnames)):
            variablesRanks[vnames[i]] = j_s[i]
        return variablesRanks
    
    def doClustering(self, k = 4):
        self.dataset.cluster_projections(k)
        return {str(clusterLabel.item()): self.dataset._clusters[clusterLabel] for clusterLabel in list(self.dataset._clusters.keys())}
    
    
    def mdsProjection(self, variables):
        alphas = self.dataSettings[SETTINGS_ALPHAS]
        
        self.dataset.compute_distance_matrix(variables=variables,
                                             alphas=alphas,
                                             distanceType= DistanceType.EUCLIDEAN,
                                             )
        self.dataset.compute_projection()
        
        self.D_k = self.dataset.distanceMatrix_k
        
        # D, self.D_k = euclidean_distance_matrix(list(self.dataset._timeSeries.values()), self.getVariablesNames(), alphas)
        # D, self.D_k = mp_distance_matrix(list(self.dataset._timeSeries.values()), self.getVariablesNames(), alphas, 12)
        self.D_k = np.power(self.D_k, 2)
        
        
        coords = [self.dataset._projections[id] for id in self.dataset.ids]
        
        coords = np.array(coords)
        print(coords)
        print(coords.shape)
        
        
        if isinstance(self.oldCoords, np.ndarray): 
            P = coords
            Q = self.oldCoords
            A = P.transpose().dot(Q)
            u, s, vt = np.linalg.svd(A, full_matrices=True)
            v = vt.transpose()
            ut = u.transpose()
            r = np.sign(np.linalg.det(v.dot(ut)))
            R = v.dot(np.array([[1, 0], [0, r]])).dot(ut)
            coords = R.dot(P.transpose()).transpose()
        
        coordsDict = {}
        ids = self.dataset.ids
        for i in range(len(ids)):
            id = ids[i]
            coord = coords[i]
            coordsDict[id] = coord.tolist()
        
        self.oldCoords = coords
        
        return coordsDict