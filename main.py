from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn import cluster
from mts.core.mtserie_dataset import MTSerieDataset
from mts.core.projections import ProjectionAlg
from models.emotion_dataset_controller import *

import json
import numpy as np
from flask import jsonify

app = Flask(__name__)
CORS(app)

appController = AppController()


@app.route('/')
def index():
    return "<h1>Welcome to our server !!</h1>"


@app.route("/routeGetDatasetsInfo", methods=['POST'])
def getDatasetsInfo():
    return jsonify({
        "loadedDatasetsIds": appController.loadedDatasets,
        "localDatasetsIds": appController.localDatasetsIds,
    })


@app.route("/loadLocalDataset", methods=['POST'])
def loadLocalDataset():
    datasetId = request.form.get('datasetId')
    succed = appController.loadLocalDataset(datasetId)
    return jsonify({
        "state": "success" if succed else "error"
    })


@app.route("/removeDataset", methods=['POST'])
def removeDataset():
    datasetId = request.form.get('datasetId')
    succed = appController.removeDataset(datasetId)
    return jsonify({
        "state": "success" if succed else "error"
    })


@app.route("/initializeDataset", methods=['POST'])
def initializeDataset():
    datasetInfoJson = request.form.get('datasetInfo')
    datasetId = appController.initializeDataset(datasetInfoJson)
    print(f"Initialized: {datasetId}")
    return jsonify({
        "id": datasetId
    })


@app.route("/addEmlToDataset", methods=['POST'])
def addEmlToDataset():
    eml = request.form.get('eml')
    datasetId = request.form.get('datasetId')
    succed = appController.addEmlToDataset(datasetId, eml)
    return jsonify({
        "state": "success" if succed else "error"
    })


@app.route("/getDatasetInfo", methods=['POST'])
def getDatasetInfo():
    datasetId = request.form.get('datasetId')
    dataInfo = appController.getDatasetInfo(datasetId)
    return jsonify(dataInfo)


@app.route("/getMTSeries", methods=['POST'])
def getMTSeries():
    datasetId = request.form.get('datasetId')
    begin = int(request.form.get('begin'))
    end = int(request.form.get('end'))
    ids = json.loads(request.form.get('ids'))
    saveEmotions = request.form.get('saveEmotions', type=int)
    if saveEmotions == 1:
        saveEmotions = True
    else:
        saveEmotions = False

    return jsonify(
        appController.getMTSeriesInRange(
            datasetId, ids, begin, end, saveEmotions=saveEmotions)
    )


@app.route("/downsampleData", methods=['POST'])
def downsampleData():
    datasetId = request.form.get('datasetId')
    rule = request.form.get('rule')
    appController.downsampleDataset(datasetId, rule)
    return jsonify({
        "state": "success"
    })


@app.route("/resetDataset", methods=['POST'])
def resetDataset():
    datasetId = request.form.get('datasetId')
    appController.resetDataset(datasetId)
    return jsonify({
        "state": "success"
    })


"""
    distance: 0 for euclidean, 1 for DTW, 2 for MPDist
"""


@app.route("/getDatasetProjection", methods=['POST'])
def getDatasetProjection():
    datasetId = request.form.get('datasetId')
    begin = request.form.get('begin', type=int)
    end = request.form.get('end', type=int)
    alphas = request.form.get('alphas', type=dict)
    distance = request.form.get('distance', type=int)
    projection = request.form.get('projection', type=int)
    projectionParameter = request.form.get('projectionParameter', type=int)
    distance = request.form.get('distance', type=int)
    D_k = json.loads(request.form.get('D_k'))
    oldCoords = json.loads(request.form.get('oldCoords'))
    if len(D_k) == 0:
        D_k = None
    else:
        D_k = {key: np.array(D_k[key]) for key in D_k.keys()}
    if len(oldCoords) == 0:
        oldCoords = None
    else:
        oldCoords = np.array(list(oldCoords.values()))
    alphas = json.loads(request.form.get('alphas'))

    distanceType = DistanceType(distance)
    projectionAlg = ProjectionAlg(projection)
    coords, D_k = appController.getProjection(
        datasetId,
        begin, end,
        alphas,
        oldCoords=oldCoords,
        D_k=D_k,
        distanceType=distanceType,
        projectionAlg=projectionAlg,
        projectionParam=projectionParameter
    )
    D_k = {key: D_k[key].tolist() for key in D_k.keys()}
    return jsonify({'coords': coords, 'D_k': D_k})


@app.route("/kmeansClustering", methods=['POST'])
def kmeansClustering():
    datasetId = request.form.get('datasetId')
    coords = json.loads(request.form.get('coords'))
    k = int(request.form.get('k'))

    clusters = appController.kmeans_clustering(
        datasetId, coords, k=k
    )
    clusters = {int(k): clusters[k] for k in clusters.keys()}

    return jsonify(clusters)


@app.route("/dbscanClustering", methods=['POST'])
def dbscanClustering():
    datasetId = request.form.get('datasetId')
    coords = json.loads(request.form.get('coords'))
    min_samples = int(request.form.get('min_samples'))
    eps = float(request.form.get('eps'))

    clusters = appController.dbscan_clustering(
        datasetId, coords, eps=eps, min_samples=min_samples
    )
    clusters = {int(k): clusters[k] for k in clusters.keys()}

    # removing outliers
    clusters.pop(-1, None)
    return jsonify(clusters)


@app.route("/getFishersDiscriminantRanking", methods=['POST'])
def getFishersDiscriminantRanking():
    datasetId = request.form.get('datasetId')
    D_k = json.loads(request.form.get('D_k'))
    D_k = {key: np.array(D_k[key]) for key in D_k.keys()}
    blueCluster = json.loads(request.form.get('blueCluster'))
    redCluster = json.loads(request.form.get('redCluster'))

    j_s = appController.getFishersDiscriminantRanking(
        datasetId, D_k, blueCluster, redCluster)

    return jsonify(j_s)


@app.route("/getTemporalSummary", methods=['POST'])
def getTemporalSummary():
    datasetId = request.form.get('datasetId')
    temporalSummary = appController.getTemporalSummary(datasetId)
    return jsonify(temporalSummary)


@app.route("/getTemporalGroupSummary", methods=['POST'])
def getTemporalGroupSummary():
    datasetId = request.form.get('datasetId')
    begin = request.form.get('begin', type=int)
    end = request.form.get('end', type=int)
    ids = json.loads(request.form.get('ids'))
    temporalSummary = appController.getTemporalGroupSummary(
        datasetId, ids, begin, end)
    return jsonify(temporalSummary)


@app.route("/getInstanceGroupSummary", methods=['POST'])
def getInstanceGroupSummary():
    datasetId = request.form.get('datasetId')
    begin = request.form.get('begin', type=int)
    end = request.form.get('end', type=int)
    ids = json.loads(request.form.get('ids'))
    temporalSummary = appController.getInstanceGroupSummary(
        datasetId, ids, begin, end)
    return jsonify(temporalSummary)


@app.route("/getValenceArousalHistogram", methods=['POST'])
def getValenceArousalHistogram():
    datasetId = request.form.get('datasetId')
    begin = request.form.get('begin', type=int)
    end = request.form.get('end', type=int)
    ids = json.loads(request.form.get('ids'))
    arousal = request.form.get('arousal')
    valence = request.form.get('valence')
    histogram, maxCellCount = appController.getValenceArousalHistogram(
        datasetId, ids, begin, end, valence, arousal)
    return jsonify({'histogram': histogram, 'cellCount': maxCellCount})


if __name__ == "__main__":
    app.run()
