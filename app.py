from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn import cluster
from mts.core.mtserie_dataset import MTSerieDataset
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
    datasetId = request.form.get('datasetId')
    succed = appController.initializeDataset(datasetId)
    return jsonify({
        "state": "success" if succed else "error"
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
    return jsonify(appController.getMTSeriesInRange(datasetId, ids, begin, end))

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


@app.route("/getDatasetProjection", methods=['POST'])
def getDatasetProjection():
    datasetId = request.form.get('datasetId')
    begin = request.form.get('begin', type=int)
    end = request.form.get('end', type=int)
    alphas = request.form.get('alphas', type=dict)
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
    # TODO
    # distanceType = int(request.form.get('distanceType'))
    coords, D_k = appController.getMdsProjection(
        datasetId, begin, end, alphas, oldCoords=oldCoords, D_k=D_k)
    D_k = {key: D_k[key].tolist() for key in D_k.keys()}
    return jsonify({'coords': coords, 'D_k': D_k})


@app.route("/doClustering", methods=['POST'])
def doClustering():
    datasetId = request.form.get('datasetId')
    coords = json.loads(request.form.get('coords'))
    k = int(request.form.get('k'))

    clusters = appController.doClustering(datasetId, coords, k)
    clusters = {int(k): clusters[k] for k in clusters.keys()}
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


if __name__ == "__main__":
    app.run()
