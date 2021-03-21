import xml.etree.ElementTree as ET 
import sys
import datetime
import numpy as np
from tslearn.metrics import dtw
from dateutil import parser
import json


sys.path.append("..")
from mts.core.mtserie import MTSerie

def mtserie_from_json(jsonString)->MTSerie:
    data = json.loads(jsonString)
    variablesNames = data["vocabulary"]
    identifiers = data["info"]["identifiers"]
    
    categoricalDict = data["info"].get("categoricalMetadata", {})
    numericalDict = data["info"].get("numericalMetadata", {})
    dateTimesStr = data.get("dates", [])
    dateTimes = [ np.datetime64(parser.parse(e)) for e in dateTimesStr]
    dateTimes = np.array(dateTimes)

    
    variablesDataDict = {} 
    for variable in variablesNames:
        variablesDataDict[variable] = np.array(data[variable])
    mtserie = MTSerie.fromDict(X=variablesDataDict, index=dateTimes, 
                               info= identifiers, numericalFeatures=numericalDict,
                               categoricalFeatures=categoricalDict)
    return mtserie
   
def mtserieQueryToJsonStr(query):
    assert isinstance(query, dict)
    if isinstance(next(iter(query.values())), np.ndarray):
        newQuery = {}
        for id, series in query.items():
            newQuery[id] = series.tolist()
        return json.dumps(newQuery)
    return json.dumps(query)