from local_datasets_info import *
# from mts.core.projections import ProjectionAlg, euclidean_distance_matrix, mds_projection, compute_k_distance_matrixes, compute_distance_matrix
# from mts.core.projections import mp_distance_matrix
from mts.core.utils import mtserieQueryToJsonStr, subsetSeparationRanking, fishersDiscriminantRanking, scale_layout
# from mts.core.distances import DistanceType, ts_euclidean_distance
from sklearn.metrics.pairwise import cosine_similarity
from utils.utils import mtserie_from_json
from mts.core.mtserie_dataset import MTSerieDataset
from numpy.lib.index_tricks import CClass
from mts.core.mtserie import MTSerie
from dateutil import parser
from math import *
import datetime
import numpy as np
import sys
import json
import umap

# sys.path.insert(1, '/home/texs/Documents/Kusisqa/repositories/Time-series-classification-and-clustering-with-Reservoir-Computing/code')

sys.path.append("..")

# from modules import RC_model
from esn_ae_v2 import EsnAe2

N_EPOCHS = 2000
EMBEDDED_SIZE = 100
INTERNAL_UNITS = 100
INFO_MIN_VALUES = "globalEmotionMin"
INFO_MAX_VALUES = "globalEmotionMax"
INFO_SERIES_LABELS = "seriesLabels"
INFO_LEN_TIME = "temporalLen"
INFO_LEN_INSTANCE = "instanceLen"
INFO_LEN_VARIABLES = "variablesLen"
INFO_DATES = "dates"
INFO_LABELS = "labels"
INFO_IS_DATED = "isDated"
INFO_DOWNSAMPLE_RULES = "downsampleRules"
INFO_IDS = "ids"
INFO_IDENTIFIERS_LABELS = "identifiersLabels"
INFO_CATEGORICAL_LABELS = "categoricalLabels"
INFO_NUMERICAL_LABELS = "numericalLabels"
INFO_TYPE = "type"  # * either 'dimensional' or 'categorical'
INFO_DIMENSIONS = "dimensions"  # * only if type is dimensional
SETTINGS_EMOTIONS_LABELS = "emotionLabels"
SETTINGS_LOWER_BOUNDS = "globalEmotionLowerBound"
SETTINGS_UPPER_BOUNDS = "globalEmotionUpperBound"

DATASET_PATH = "datasets/"


def rangeConverter(oldValue, oldMin, oldMax,
                   newMin, newMax):
    return (((oldValue - oldMin) * (newMax - newMin)) / (oldMax - oldMin)) + newMin


rc_config = {}

# * Reservoir cnn_settings
rc_config['n_internal_units'] = 50        # size of the reservoir
rc_config['spectral_radius'] = 0.59        # largest eigenvalue of the reservoir
rc_config['leak'] = 0.6                    # amount of leakage in the reservoir state update (None or 1.0 --> no leakage)
rc_config['connectivity'] = 0.35           # percentage of nonzero connections in the reservoir
rc_config['input_scaling'] = 0.1           # scaling of the input weights
rc_config['noise_level'] = 0.01            # noise in the reservoir state update
rc_config['n_drop'] = 5                    # transient states to be dropped
rc_config['bidir'] = True                  # if True, use bidirectional reservoir
rc_config['circ'] = False                  # use reservoir with circle topology

# Dimensionality reduction
rc_config['dimred_method'] ='tenpca'       # options: {None (no dimensionality reduction), 'pca', 'tenpca'}
# rc_config['dimred_method'] = 'None'
# rc_config['n_dim'] = 10                   # number of resulting dimensions after the dimensionality reduction procedure
rc_config['n_dim'] = 10
# MTS representation
rc_config['mts_rep'] = 'reservoir'         # MTS representation:  {'last', 'mean', 'output', 'reservoir'}
# rc_config['mts_rep'] = 'reservoir'
rc_config['w_ridge_embedding'] = 10.0      # regularization parameter of the ridge regression

# Readout
rc_config['readout_type'] = 'svm'           # by setting None, the input representations will be stored
rc_config['readout_type'] = None
class AppController:
    def __init__(self):
        self.loadedDatasets = []
        self.localDatasetsIds = [
            # "motions",
            # "wafer",
            # "libras",
            # "uwave",
            # "stand",
            # "handwritting",
            "AMIGOS_dimensional",
            "wesad",
            "case",
            "aff-wild-categorical",
            "aff-wild-dimensional",
            "ascertain",
            "drivers_workload",
            "drivers_stress",
            "emotions_in_music",
            "afew_va",
            "wesad_dimensional_3",
            "wesad_categorical_panas",
            "wesad_categorical_stai",
            # "case_categorical",
        ]
        self.datasets = {}
        # this info is changed according to the proccesing make on the original data
        self.datasetsInfo = {}
        self.mts_representations = {}

    def loadLocalDataset(self, datasetId):
        if datasetId in self.loadedDatasets:
            return False
        
        if datasetId == "wesad_dimensional_3":
            path_info = wesad_path_info_dimensional_3
            paths = wesad_paths

        elif datasetId == "wesad":
            path_info = wesad_path_info_dimensional_2
            paths = wesad_paths
        elif datasetId == "wesad_categorical_panas":
            path_info = wesad_path_info_categorical_panas
            paths = wesad_paths
        elif datasetId == "wesad_categorical_stai":
            path_info = wesad_path_info_categorical_stai
            paths = wesad_paths
        # elif datasetId == "case_dimensional":
        #     path_info = case_path_info_dimensional
        #     paths = case_paths
        elif datasetId == "case":
            path_info = case_path_info_categorical
            paths = case_paths
        elif datasetId == "ascertain":
            path_info = ascertain_path_info
            paths = ascertain_paths
        elif datasetId == "drivers_workload":
            path_info = workload_path_info
            paths = workload_paths
        elif datasetId == "drivers_stress":
            path_info = stress_path_info
            paths = stress_paths
        elif datasetId == "emotions_in_music":
            path_info = emotion_in_music_path_info
            paths = emotion_in_music_paths
        elif datasetId == "afew_va":
            path_info = afew_va_path_info
            paths = afew_va_paths
        elif datasetId == "aff-wild-categorical":
            path_info = aff_wild_categorical_path_info
            paths = aff_wild_categorical_paths
        elif datasetId == "aff-wild-dimensional":
            path_info = aff_wild_dimensional_path_info
            paths = aff_wild_dimensional_paths
        elif datasetId == "AMIGOS_dimensional":
            path_info = amigos_dimensional_path_info
            paths = amigos_dimensional_paths
        
        # elif datasetId == "motions":
        #     path_info = motions_path_info
        #     paths = motions_paths
        # elif datasetId == "wafer":
        #     path_info = wafer_path_info
        #     paths = wafer_paths
        # elif datasetId == "libras":
        #     path_info = libras_path_info
        #     paths = libras_paths
        # elif datasetId == "uwave":
        #     path_info = uwave_path_info
        #     paths = uwave_paths
        # elif datasetId == "stand":
        #     path_info = stand_path_info
        #     paths = stand_paths
        # elif datasetId == "handwritting":
        #     path_info = handwritting_path_info
        #     paths = handwritting_paths

        with open('datasets/' + path_info, 'r') as file:
            dataInfoJson = file.read()
        self.initializeDataset(dataInfoJson)

        for path in paths:
            with open(DATASET_PATH + path, 'r') as file:
                jsonStr = file.read()
                self.addMtserieFromString(datasetId, jsonStr)

        return True

    def addMtserieFromString(self, datasetId, eml):
        datasetInfo = self.datasetsInfo[datasetId]
        mtserie = None
        if "dates" in datasetInfo:
            dateTimes = datasetInfo["dates"]
            mtserie = mtserie_from_json(eml, dateTimes=dateTimes)
        else:
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
            return "error"
        self.datasets[datasetId] = MTSerieDataset()
        self.datasetsInfo[datasetId] = infoDict
        self.loadedDatasets.append(datasetId)
        if "dates" in infoDict:
            dateTimes = [np.datetime64(parser.parse(e))
                         for e in infoDict["dates"]]
            dateTimes = np.array(dateTimes)
            infoDict["dates"] = dateTimes
        return datasetId

    def addEmlToDataset(self, datasetId, eml):
        if not datasetId in self.loadedDatasets:
            return False
        id = self.addMtserieFromString(datasetId, eml)
        return True

    def getMTSeriesInRange(self, datasetId, ids, begin, end, saveEmotions: bool = False):
        query = self.datasets[datasetId].get_mtseries_in_range(
            begin, end, ids, procesed=True)
        resultMap = {id: self.mtserieToMap(
            query[id], saveEmotions=saveEmotions) for id in list(query.keys())}
        return resultMap

    def downsampleDataset(self, datasetId, rule):
        self.datasets[datasetId].downsample_data(rule)

    def mtserieToMap(self, mtserie, saveEmotions: bool = False):
        assert isinstance(mtserie, MTSerie)
        mtserieMap = {}
        if saveEmotions:
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
            dataInfo[INFO_CATEGORICAL_LABELS] = self.datasetsInfo[datasetId]["vocabulary"]['categoricalMetadata']
        if 'numericalMetadata' in self.datasetsInfo[datasetId]["vocabulary"]:
            dataInfo[INFO_NUMERICAL_LABELS] = self.datasetsInfo[datasetId]["vocabulary"]['numericalMetadata']

        dataInfo[INFO_IDENTIFIERS_LABELS] = self.datasetsInfo[datasetId]["vocabulary"]["identifiers"]

        if "labels" in self.datasetsInfo[datasetId]:
            dataInfo[INFO_LABELS] = self.datasetsInfo[datasetId]["labels"]

        if self.datasets[datasetId].isDataDated:
            dataInfo[INFO_DATES] = [
                str(date) for date in self.datasets[datasetId].get_datetimes(procesed=True)]
            dataInfo[INFO_DOWNSAMPLE_RULES] = self.datasets[datasetId].allowedDownsampleRules
        return dataInfo

    # ! deprecated
    # def compute_D_k(self, datasetId, begin, end, variables, distanceType=DistanceType.EUCLIDEAN):
    #     D_k = compute_k_distance_matrixes(
    #         list(self.datasets[datasetId].get_mtseries_in_range(
    #             begin, end).values()),
    #         variables,
    #         distanceType
    #     )
    #     return D_k

    def getProjection(
        self,
        datasetId,
        begin,
        end,
        oldCoords=None,
        projectionParam: int = 5
    ):

        n_internal_units = INTERNAL_UNITS
        spectral_radius=0.9
        leak=None
        connectivity=0.35
        input_scaling=0.1
        noise_level=0.01
        circle=False
        n_epochs = N_EPOCHS
        embedding_size = EMBEDDED_SIZE

        X = self.datasets[datasetId].values()

        X = X [:, begin:end,:]
        N, T, D = X.shape

        ae = EsnAe2(
            n_internal_units=n_internal_units,
            spectral_radius=spectral_radius,
            leak=leak,
            connectivity=connectivity,
            input_scaling=input_scaling,
            noise_level=noise_level,
            circle=circle,
            n_drop=0, 
            bidir=False,
            pca_n_dim = None,
            n_epochs = n_epochs,
            w_out_mode= False,
            embedding_size= embedding_size
        )

        mts_representations = ae.train(X)

        similarity_matrix = cosine_similarity(mts_representations)
        similarity_matrix = (similarity_matrix + 1.0)/2.0

        D = similarity_matrix

        reducer = umap.UMAP(n_neighbors=projectionParam)
        coords = reducer.fit_transform(mts_representations)

        coords = scale_layout(coords)

        # * return coords as dict
        coordsDict = {}
        ids = self.datasets[datasetId].ids
        for i in range(len(ids)):
            id = ids[i]
            coordsDict[id] = coords[i].tolist()
        
        # *  set projections in dataset
        for i in range(self.datasets[datasetId].instanceLen):
            self.datasets[datasetId]._projections[self.datasets[datasetId].ids[i]] = coords[i]
        
        self.mts_representations[datasetId] = mts_representations;
        return coordsDict
        

    def dbscan_clustering(self, datasetId, coords, eps=0.2, min_samples=10):
        # def doClustering(self, datasetId, coords, k=4):
        clusters = self.datasets[datasetId].cluster_projections_dbscan(
            coords, eps=eps, min_samples=min_samples
        )
        return clusters

    def kmeans_clustering(self, datasetId, coords, k=4):
        clusters = self.datasets[datasetId].cluster_projections_kmeans(
            k, coords
        )
        return clusters

    def getFishersDiscriminantRanking(self, datasetId, blueCluster, redCluster):
        dataset = self.datasets[datasetId]
        variables = dataset.temporalVariables
        variablesRanks = {}
        ids = dataset.ids
        representations = self.mts_representations[datasetId]
        D = dataset.variablesLen
        u_ind = [ids.index(e) for e in blueCluster]
        v_ind = [ids.index(e) for e in redCluster]

        varReprLen = int(len(representations[0]) // D)

        for i in range(D):
            print(representations.shape)
            # print(representations)
            varRepr = representations[:, i * varReprLen : (i + 1) * varReprLen ]
            variable = variables[i]

            u = varRepr[u_ind]
            v = varRepr[v_ind]

            u_mean = np.mean(u, axis=0)
            v_mean = np.mean(v, axis=0)

            u_var = np.var(u, axis=0)
            v_var = np.var(v, axis=0)

            # print(f"u: {u}")
            # print(f"v: {v}")
            print(f"u_m: {u_mean}")
            print(f"v_m: {v_mean}")
            print(f"u_s: {u_var}")
            print(f"v_s: {v_var}")
            print(f"repr shape: {representations.shape}")
            print(f"u shape: {u.shape}")
            print(f"v shape: {v.shape}")
            print(u_mean.shape)
            print(v_mean.shape)

            # variablesRanks[variable] = np.linalg.norm(u_mean - v_mean) / (np.linalg.norm(u_var) + np.linalg.norm(v_var))
            variablesRanks[variable] = i * 0.1
            print(f"rank: {variable} -> {variablesRanks[variable]}")

            # variablesRanks[variable] = j_s[varName]
        print(variablesRanks)




        # j_s = fishersDiscriminantRanking(D_ks, blueIndexes, redIndexe
        # s)
        # variables = list(D_ks.keys())
        # for varName in variables:
        #     variablesRanks[varName] = j_s[varName]
        return variablesRanks

    def getTemporalGroupSummary(
        self,
        datasetId,
        ids,
        begin,
        end
    ):
        mtseries = self.datasets[datasetId].get_mtseries_in_range(
            begin,
            end,
            ids,
            procesed=True
        )
        variables = self.datasets[datasetId].temporalVariables
        emotions = np.array([[mtseries[p_id].get_serie(var)
                            for var in variables] for p_id in ids])
        emotions = np.transpose(emotions, (1, 0, 2))
        summary = {}
        summary['min'] = {}
        for i in range(len(variables)):
            summary['min'][variables[i]] = emotions[i].min(axis=0).tolist()

        summary['max'] = {}
        for i in range(len(variables)):
            summary['max'][variables[i]] = emotions[i].max(axis=0).tolist()

        summary['mean'] = {}
        for i in range(len(variables)):
            summary['mean'][variables[i]] = emotions[i].mean(axis=0).tolist()

        summary['std'] = {}
        for i in range(len(variables)):
            summary['std'][variables[i]] = emotions[i].std(axis=0).tolist()

        return summary

    def getInstanceGroupSummary(
        self,
        datasetId,
        ids,
        begin,
        end
    ):
        mtseries = self.datasets[datasetId].get_mtseries_in_range(
            begin,
            end,
            ids,
            procesed=True
        )
        variables = self.datasets[datasetId].temporalVariables
        emotions = np.array([[mtseries[p_id].get_serie(var)
                            for var in variables] for p_id in ids])
        emotions = np.transpose(emotions, (1, 0, 2))
        summary = {}
        summary['min'] = {}
        for i in range(len(variables)):
            summary['min'][variables[i]] = emotions[i].min()

        summary['max'] = {}
        for i in range(len(variables)):
            summary['max'][variables[i]] = emotions[i].max()

        summary['mean'] = {}
        for i in range(len(variables)):
            summary['mean'][variables[i]] = emotions[i].mean()

        summary['std'] = {}
        for i in range(len(variables)):
            summary['std'][variables[i]] = emotions[i].std()

        return summary

    def getInstanceGroupSummary(
        self,
        datasetId,
        ids,
        begin,
        end
    ):
        mtseries = self.datasets[datasetId].get_mtseries_in_range(
            begin,
            end,
            ids,
            procesed=True
        )
        variables = self.datasets[datasetId].temporalVariables
        emotions = np.array([[mtseries[p_id].get_serie(var)
                            for var in variables] for p_id in ids])
        emotions = np.transpose(emotions, (1, 0, 2))
        summary = {}
        summary['min'] = {}
        for i in range(len(variables)):
            summary['min'][variables[i]] = emotions[i].min()

        summary['max'] = {}
        for i in range(len(variables)):
            summary['max'][variables[i]] = emotions[i].max()

        summary['mean'] = {}
        for i in range(len(variables)):
            summary['mean'][variables[i]] = emotions[i].mean()

        summary['std'] = {}
        for i in range(len(variables)):
            summary['std'][variables[i]] = emotions[i].std()

        return summary

    def getValenceArousalHistogram(
        self,
        datasetId,
        ids,
        begin,
        end,
        valenceVar,
        arousalVar
    ):
        mtseries = self.datasets[datasetId].get_mtseries_in_range(
            begin,
            end,
            ids,
            procesed=True
        )
        n_sideBins: int = 20
        valenceEmotions = np.array([mtseries[p_id].get_serie(valenceVar)
                                    for p_id in ids]).flatten()
        arousalEmotions = np.array([mtseries[p_id].get_serie(arousalVar)
                                    for p_id in ids]).flatten()
        histogram = np.zeros([n_sideBins, n_sideBins])
        minValence = self.getDatasetEmotionValues(datasetId, "min")[valenceVar]
        maxValence = self.getDatasetEmotionValues(datasetId, "max")[valenceVar]
        minArousal = self.getDatasetEmotionValues(datasetId, "min")[arousalVar]
        maxArousal = self.getDatasetEmotionValues(datasetId, "max")[arousalVar]
        maxCellCount = 0
        for i in range(len(valenceEmotions)):
            x = floor(rangeConverter(
                valenceEmotions[i],
                minValence,
                maxValence,
                0.0,
                float(n_sideBins - 1)
            ))
            y = floor(rangeConverter(
                arousalEmotions[i],
                minArousal,
                maxArousal,
                0.0,
                (n_sideBins - 1),
            ))

            histogram[x][y] = histogram[x][y] + 1
            if (histogram[x][y] > maxCellCount):
                maxCellCount = histogram[x][y]

        histogram = histogram.flatten()
        return histogram.tolist(), maxCellCount

    def getGroupSummary(
        self,
        datasetId,
        ids,
        begin,
        end
    ):
        mtseries = self.datasets[datasetId].get_mtseries_in_range(
            begin,
            end,
            ids,
            procesed=True
        )
        variables = self.datasets[datasetId].temporalVariables
        emotions = np.array([[mtseries[p_id].get_serie(var)
                            for var in variables] for p_id in ids])
        emotions = np.transpose(emotions, (1, 0, 2))
        summary = {}
        summary['min'] = {}
        for i in range(len(variables)):
            summary['min'][variables[i]] = emotions[i].min(axis=0).tolist()

        summary['max'] = {}
        for i in range(len(variables)):
            summary['max'][variables[i]] = emotions[i].max(axis=0).tolist()

        summary['mean'] = {}
        for i in range(len(variables)):
            summary['mean'][variables[i]] = emotions[i].mean(axis=0).tolist()

        summary['std'] = {}
        for i in range(len(variables)):
            summary['std'][variables[i]] = emotions[i].std(axis=0).tolist()

        return summary

    def getTemporalSummary(self, datasetId):
        values = self.datasets[datasetId].values()
        allVariables = self.datasets[datasetId].temporalVariables
        values = np.transpose(values, (2, 1, 0))  # to D, T, N shape
        summary = {}
        emotions = self.getDatasetEmotions(datasetId)
        summary['min'] = {}
        emotion_pos = [allVariables.index(emotions[i])
                       for i in range(len(emotions))]
        for i in range(len(emotions)):
            pos = emotion_pos[i]
            summary['min'][emotions[i]] = values[pos].min(axis=1).tolist()

        summary['max'] = {}
        for i in range(len(emotions)):
            pos = emotion_pos[i]
            summary['max'][emotions[i]] = values[pos].max(axis=1).tolist()

        summary['mean'] = {}
        for i in range(len(emotions)):
            pos = emotion_pos[i]
            summary['mean'][emotions[i]] = values[pos].mean(axis=1).tolist()

        return summary

    def resetDataset(self, datasetId):
        self.datasets[datasetId].resetProcesedMtseries()
