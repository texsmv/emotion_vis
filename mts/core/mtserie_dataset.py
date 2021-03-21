import numpy as np
from .mtserie import MTSerie
from numpy import unique
from .distances import DistanceType
from .projections import distance_matrix, mds_projection
from sklearn.cluster import SpectralClustering, KMeans, DBSCAN

class MTSerieDataset:
    """summary for [MTSerieDataset]
    
        This class assumes that the [MTSerie] objects use the same
        variables, although they can be uneven or misaligned with 
        respect to time
    """
    @property
    def allowedDownsampleRules(self)->list:
        return self.first.downsample_rules()
    
    @property
    def isDataDated(self) -> bool:
        return self.first.isDataDated
    
    def get_datetimes(self, procesed = True) -> np.array:
        return self.get_first(procesed).datetimes
    datetimes = property(get_datetimes)
    
    @property
    def distanceMatrix(self) -> np.ndarray:
        return self._distanceMatrix
    
    @distanceMatrix.setter
    def distanceMatrix(self, value):
        self._distanceMatrix = value
    
    @property
    def distanceMatrix_k(self) -> list:
        return self._distanceMatrix_k
    
    @distanceMatrix_k.setter
    def distanceMatrix(self, value):
        self._distanceMatrix_k = value
    
    @property
    def ids(self) -> list:
        return list(self.mtseries.keys())
    
    
    def get_first(self, procesed = True) -> MTSerie:
        if procesed:
            return self.procesedMTSeries[next(iter(self.procesedMTSeries))]
        else:
            return self.mtseries[next(iter(self.mtseries))]
    first = property(get_first)
    
    @property
    def temporalVariables(self):
        if self._isDataUniformInVariables:
            return self.first.labels
        return [mtserie.labels for mtserie in self.mtseries.values()]
    
    @property
    def variablesLen(self) -> int:
        if self._isDataUniformInVariables:
            return self.first.variablesLen
        return [mtserie.variablesLen for mtserie in self.mtseries.values()]
    
    def get_timeLen(self,procesed = True) -> int:
        if self._isDataUniformInTime:
            return self.get_first(procesed).timeLen
        return [mtserie.timeLen for mtserie in self.get_mtseries(procesed=procesed)]
    timeLen = property(get_timeLen)    
    
    @property
    def instanceLen(self):
        return len(self.mtseries)
    
    @property
    def categoricalLabels(self) -> list:
        return self.first.categoricalLabels
    
    @property
    def numericalLabels(self) -> list:
        return self.first.numericalLabels
    
    def __init__(self):
        self.mtseries = {}
        self.procesedMTSeries = {}
        self._isDataUniformInTime = True
        self._isDataUniformInVariables = True
        self._distanceMatrix = None
        self._distanceMatrix_k = None
        self.oldCoords = None

        
        self.d_k = {}
        self._variablesLimits = {}
        self._projections = {}
        self._clusters = {}
        self._clusterById = {}
        self.minTemporalValues = {}
        self.maxTemporalValues = {}
        
        super().__init__()

    def add(self, mtserie, identifier):
        assert isinstance(mtserie, MTSerie)
        assert isinstance(identifier, str)
        
        self.mtseries[identifier] = mtserie
        # * Added to procesed mtseries by reference
        self.procesedMTSeries[identifier] = mtserie
        
        if self._isDataUniformInVariables:
            self._isDataUniformInVariables = self.variablesLen == mtserie.variablesLen 
        
        if self._isDataUniformInTime:
            self._isDataUniformInTime = self.timeLen == mtserie.timeLen
        
        mtserieMins = mtserie.minValues
        mtserieMaxs = mtserie.maxValues
        
        if len(self.minTemporalValues) == 0 or len(self.maxTemporalValues) == 0:
            self.minTemporalValues = mtserieMins
            self.maxTemporalValues = mtserieMaxs
        else:        
            for varName in self.temporalVariables:
                if mtserieMins[varName] < self.minTemporalValues[varName]:
                    self.minTemporalValues[varName] = mtserieMins[varName]
                if mtserieMaxs[varName] > self.maxTemporalValues[varName]:
                    self.maxTemporalValues[varName] = mtserieMaxs[varName]
        
        assert self.categoricalLabels == mtserie.categoricalLabels
        assert self.numericalLabels == mtserie.numericalLabels
    
    
    
    def get_mtseries(self, procesed = True, ids = []):
        if len(ids) == 0:
            if procesed:
                return list(self.procesedMTSeries.values())
            else:
                return list(self.mtseries.values())
        else:
            if procesed:
                return [self.procesedMTSeries[id] for id in ids]
            else:
                return [self.mtseries[id] for id in ids]
    
    def get_mtserie(self, id, procesed = True)->MTSerie:

        if procesed:
            return self.procesedMTSeries[id]
        else:
            return self.mtseries[id]
    
    def compute_k_distance_matrix(self, variables = [], alphas = {}, distanceType = DistanceType.EUCLIDEAN, L = 10, procesed = True):
        _variables = variables
        if len(variables) == 0: 
            _variables = self.temporalVariables
        
        _alphas = alphas
        if len(alphas) == 0:
            _alphas = {var: 1.0 for var in variables}
            
        assert len(_alphas) == len(_variables)
    
        self._distanceMatrix, self._distanceMatrix_k = distance_matrix(
            self.get_mtseries(procesed=procesed), variables=_variables, 
            alphas=_alphas, distanceType=distanceType, L=L
            )
    
    def compute_distance_matrix(self, variables = [], alphas = [], distanceType = DistanceType.EUCLIDEAN, L = 10, procesed = True):
        '''
        Implementation of distance matrix defined in "Interactive visualization of multivariate time series data"

        Args:
            distanceType (DistanceType, optional): Distance to compare mtseries. Defaults to DistanceType.EUCLIDEAN.
            L (int, optional): Window size used for MPdist. Defaults to 10.
        '''
        _variables = variables
        if len(variables) == 0: 
            _variables = self.temporalVariables
        
        _alphas = alphas
        if len(alphas) == 0:
            _alphas = np.ones(len(_variables))
        assert len(_alphas) == len(_variables)
    
        self._distanceMatrix, self._distanceMatrix_k = distance_matrix(
            self.get_mtseries(procesed=procesed), variables=_variables, 
            alphas=_alphas, distanceType=distanceType, L=L
            )
    
    
    # def compute_projection(self):
    #     coords = mds_projection(self._distanceMatrix)
    #     for i in range(self.instanceLen):
    #         self._projections[self.ids[i]] = coords[i]
            
    def compute_projection(self, D):
        coords = mds_projection(D)
        for i in range(self.instanceLen):
            self._projections[self.ids[i]] = coords[i]
            
    def downsample_data(self, rule):
        for i in range(self.instanceLen):
            self.procesedMTSeries[self.ids[i]] = self.mtseries[self.ids[i]].resample(rule)
            
    def cluster_projections(self, n_clusters, coords):
        coords = np.array(list(self._projections.values()))
        
        # ! spectral clustering not working
        # clustering = SpectralClustering(n_clusters=40,
        #                         assign_labels="discretize",
        #                         n_neighbors=2,
        #                         random_state=0).fit(coords)
        # return clustering.labels_

        # * dbscan
        # print(coords.shape)
        # clustering = DBSCAN(eps=0.1, min_samples=2).fit(coords)
        # fit model and predict clusters
        # clusters = clustering.labels_
        
        k_means = KMeans(random_state=0, n_clusters=n_clusters)
        k_means.fit(coords)
        labels = k_means.predict(coords)
        
        clusters = {}
        clusterLabels = np.unique(labels)
        for clusterLabel in clusterLabels:
            clusterIds = []
            for i in range(self.instanceLen):
                if labels[i] == clusterLabel:
                    clusterIds = clusterIds + [self.ids[i]]
                    # self._clusterById[self.ids[i]] = clusterLabel
            clusters[clusterLabel] = clusterIds
        return clusters

        
    # def cluster_projections(self, n_clusters):
    #     coords = np.array(list(self._projections.values()))
        
    #     # ! spectral clustering not working
    #     # clustering = SpectralClustering(n_clusters=40,
    #     #                         assign_labels="discretize",
    #     #                         n_neighbors=2,
    #     #                         random_state=0).fit(coords)
    #     # return clustering.labels_

    #     # * dbscan
    #     # print(coords.shape)
    #     # clustering = DBSCAN(eps=0.1, min_samples=2).fit(coords)
    #     # fit model and predict clusters
    #     # clusters = clustering.labels_
        
    #     k_means = KMeans(random_state=0, n_clusters=n_clusters)
    #     k_means.fit(coords)
    #     labels = k_means.predict(coords)
        
    #     self._clusters = {}
    #     clusterLabels = np.unique(labels)
    #     for clusterLabel in clusterLabels:
    #         clusterIds = []
    #         for i in range(self.instanceLen):
    #             if labels[i] == clusterLabel:
    #                 clusterIds = clusterIds + [self.ids[i]]
    #                 self._clusterById[self.ids[i]] = clusterLabel
    #         self._clusters[clusterLabel] = clusterIds

    
    
    def query_all_by_range(self, begin, end):
        #  todo check this
        # assert self._isDataUniformInTime
        result = {}
        for id, mtserie in self.procesedMTSeries.items():
            assert isinstance(mtserie, MTSerie)
            result[id] = mtserie.range_query(begin, end)
        return result

    def get_mtseries_in_range(self, begin, end, ids = [], procesed = True)->list:
        _ids = ids
        if len(_ids) == 0:
            _ids = self.ids
        result = {}
        for id in _ids:
            mtserie = self.get_mtserie(id, procesed=procesed)
            result[id] = mtserie.range_query(mtserie.index[begin], mtserie.index[end])
        return result
    
    # ! deprecated
    def getAllMetadata(self):
        result = {}
        for id, mtserie in self._timeSeries.items():
            assert isinstance(mtserie, MTSerie)
            result[id] = {'metadata': mtserie.metadata, 'numFeatures' : mtserie.numericalFeatures.tolist(), 'numLabels' : mtserie.numericalLabels, 'catFeatures' : mtserie.categoricalFeatures.tolist(), 'catLabels' : mtserie.categoricalLabels}
        return result
    
    # ! deprecated
    def computeVariablesLimits(self):
        self._variablesLimits = {}
        for varName in self._variablesNames:
            currMin = None
            currMax = None
            for mtserie in self._timeSeries.values():
                assert isinstance(mtserie, MTSerie)
                minValue = mtserie.getSerie(varName).min()
                maxValue = mtserie.getSerie(varName).max()
                
                if(currMin == None):
                    currMin = minValue
                elif currMin > minValue:
                    currMin = minValue
                    
                if(currMax == None):
                    currMax = maxValue
                elif currMax < maxValue:
                    currMax = maxValue
            self._variablesLimits[varName] = [currMin, currMax]
            
    def get_datetime_common_range(self, procesed = True):
        begin_dates = []
        last_dates = []
        for mtserie in self.get_mtseries(procesed= procesed):
            assert isinstance (mtserie, MTSerie)
            assert mtserie.isDataDated
            begin_dates = begin_dates + [mtserie.datetimes[0]]
            last_dates = last_dates + [mtserie.datetimes[-1]]
        return (max(begin_dates), min(last_dates))
    # ! deprecated
    def getVariablesLimits(self):
        return self._variablesLimits
    # ! deprecated
    def getVariableLimits(self, varName):
        return self._variablesLimits[varName]
    # ! deprecated
    def setVariableLimits(self, varName, minValue, maxValue):
        self._variablesLimits[varName] = [minValue, maxValue]
    
    def removeVariable(self, varName):
        for mtserie in self.get_mtseries(procesed=False):
            assert isinstance(mtserie, MTSerie)
            mtserie.remove_serie(varName)
        
        for mtserie in self.get_mtseries(procesed=True):
            assert isinstance(mtserie, MTSerie)
            mtserie.remove_serie(varName)
    
    def values(self, procesed=True)-> np.ndarray:
        assert self._isDataUniformInTime
        assert self._isDataUniformInVariables
        mtseries_values = []
        for serie in self.get_mtseries(procesed=procesed):
            mtseries_values = mtseries_values + [serie.values]
        
        return np.array(mtseries_values)