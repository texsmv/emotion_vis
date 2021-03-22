def load_dataset_info(jsonString):
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
   