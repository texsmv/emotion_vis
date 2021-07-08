import 'dart:convert';

import 'package:emotion_vis_client/app_constants.dart';
import 'package:emotion_vis_client/enums/app_enums.dart';
import 'package:http/http.dart';

Future<Map<String, List<String>>> repositoryGetDatasetsInfo() async {
  final response = await post(hostUrl + "routeGetDatasetsInfo");
  final data = jsonDecode(response.body);
  return {
    "loadedDatasetsIds": List<String>.from(data["loadedDatasetsIds"]),
    "localDatasetsIds": List<String>.from(data["localDatasetsIds"]),
  };
}

Future<NotifierState> repositoryLoadLocalDataset(String datasetId) async {
  final response =
      await post(hostUrl + "loadLocalDataset", body: {"datasetId": datasetId});
  final data = jsonDecode(response.body);
  if (data["state"] == "success") {
    return NotifierState.SUCCESS;
  } else {
    return NotifierState.ERROR;
  }
}

Future<NotifierState> repositoryRemoveDataset(String datasetId) async {
  final response =
      await post(hostUrl + "removeDataset", body: {"datasetId": datasetId});
  final data = jsonDecode(response.body);
  if (data["state"] == "success") {
    return NotifierState.SUCCESS;
  } else {
    return NotifierState.ERROR;
  }
}

Future<String> repositoryInitializeDataset(String datasetInfoJson) async {
  final response = await post(hostUrl + "initializeDataset",
      body: {"datasetInfo": datasetInfoJson});
  final data = jsonDecode(response.body);
  return data["id"];
}

Future<NotifierState> repositoryAddEml(String datasetId, String eml) async {
  final response = await post(hostUrl + "addEmlToDataset", body: {
    "datasetId": datasetId,
    'eml': eml,
  });
  final data = jsonDecode(response.body);

  if (data["state"] == "success") {
    return NotifierState.SUCCESS;
  } else {
    return NotifierState.ERROR;
  }
}

Future<Map<String, dynamic>> repositoryGetDatasetInfo(String datasetId) async {
  final response = await post(hostUrl + "getDatasetInfo", body: {
    "datasetId": datasetId,
  });
  Map<String, dynamic> infoMap = jsonDecode(response.body);
  return infoMap;
}

Future<Map<String, dynamic>> repositoryGetMTSeries(
    String datasetId, int begin, int end, List<String> ids,
    {bool getEmotions = false}) async {
  var response = await post(hostUrl + "getMTSeries", body: {
    "datasetId": datasetId,
    'end': jsonEncode(end),
    'begin': jsonEncode(begin),
    'ids': jsonEncode(ids),
    'saveEmotions': jsonEncode(getEmotions ? 1 : 0)
  });
  Map<String, dynamic> queryMap = jsonDecode(response.body);
  return queryMap;
}

Future<Map<String, dynamic>> repositoryGetTemporalGroupSummary(
  String datasetId,
  int begin,
  int end,
  List<String> ids,
) async {
  var response = await post(hostUrl + "getTemporalGroupSummary", body: {
    "datasetId": datasetId,
    'end': jsonEncode(end),
    'begin': jsonEncode(begin),
    'ids': jsonEncode(ids),
  });
  Map<String, dynamic> queryMap = jsonDecode(response.body);
  return queryMap;
}

Future<Map<String, dynamic>> repositoryGetInstanceGroupSummary(
  String datasetId,
  int begin,
  int end,
  List<String> ids,
) async {
  var response = await post(hostUrl + "getInstanceGroupSummary", body: {
    "datasetId": datasetId,
    'end': jsonEncode(end),
    'begin': jsonEncode(begin),
    'ids': jsonEncode(ids),
  });
  Map<String, dynamic> queryMap = jsonDecode(response.body);
  return queryMap;
}

Future<Map<String, dynamic>> repositoryGetHistogram(String datasetId, int begin,
    int end, List<String> ids, String arousalVar, String valenceVar) async {
  var response = await post(hostUrl + "getValenceArousalHistogram", body: {
    "datasetId": datasetId,
    'end': jsonEncode(end),
    'begin': jsonEncode(begin),
    'ids': jsonEncode(ids),
    'arousal': arousalVar,
    'valence': valenceVar,
  });
  Map<String, dynamic> queryMap = jsonDecode(response.body);
  return queryMap;
}

Future<NotifierState> repositoryDownsampleData(
    String datasetId, String rule) async {
  await post(hostUrl + "downsampleData",
      body: {"datasetId": datasetId, 'rule': rule});
  return NotifierState.SUCCESS;
}

Future<NotifierState> repositoryResetDataset(
    String datasetId, String rule) async {
  await post(hostUrl + "downsampleData",
      body: {"datasetId": datasetId, 'rule': jsonEncode(rule)});
  return NotifierState.SUCCESS;
}

/// returns 'coords' and 'D_k'
Future<Map<String, dynamic>> repositoryGetDatasetProjection(
  String datasetId,
  int begin,
  int end,
  int distance,
  int projection,
  int projectionParameter,
  Map<String, double> alphas, {
  Map<String, dynamic> oldCoords = const {},
  // ignore: non_constant_identifier_names
  Map<String, dynamic> d_k = const {},
}) async {
  var response = await post(hostUrl + "getDatasetProjection", body: {
    "datasetId": datasetId,
    'end': jsonEncode(end),
    'begin': jsonEncode(begin),
    'alphas': jsonEncode(alphas),
    'D_k': jsonEncode(d_k),
    "oldCoords": jsonEncode(oldCoords),
    'distance': jsonEncode(distance),
    'projection': jsonEncode(projection),
    'projectionParameter': jsonEncode(projectionParameter)
  });
  Map<String, dynamic> queryMap = jsonDecode(response.body);
  return queryMap;
}

Future<Map<String, dynamic>> repositoryDbscanClustering(String datasetId,
    Map<String, dynamic> coords, double eps, int min_samples) async {
  var response = await post(hostUrl + "dbscanClustering", body: {
    "datasetId": datasetId,
    'coords': jsonEncode(coords),
    'eps': jsonEncode(eps),
    'min_samples': jsonEncode(min_samples),
  });
  Map<String, dynamic> clusters = jsonDecode(response.body);
  return clusters;
}

Future<Map<String, dynamic>> repositoryKmeansClustering(
    String datasetId, Map<String, dynamic> coords, int k) async {
  var response = await post(hostUrl + "kmeansClustering", body: {
    "datasetId": datasetId,
    'coords': jsonEncode(coords),
    'k': jsonEncode(k),
  });
  Map<String, dynamic> clusters = jsonDecode(response.body);
  return clusters;
}

Future<List<String>> repositoryGetFishersDiscriminantRanking(
  String datasetId,
  Map<String, dynamic> d_k,
  List<String> clusterAIds,
  List<String> clusterBIds,
) async {
  var response = await post(hostUrl + "getFishersDiscriminantRanking", body: {
    "datasetId": datasetId,
    'D_k': jsonEncode(d_k),
    'blueCluster': jsonEncode(clusterAIds),
    'redCluster': jsonEncode(clusterBIds)
  });

  Map<String, dynamic> ranks = jsonDecode(response.body);
  ranks = Map<String, double>.from(ranks);
  Map<double, String> reversedMap = {};

  for (var varName in ranks.keys) {
    reversedMap[ranks[varName]] = varName;
  }

  List<double> rankList = List.from(ranks.values.toList());
  rankList.sort((b, a) => a.compareTo(b));

  List<String> orderedVariablesNames = [];
  for (var i = 0; i < rankList.length; i++) {
    orderedVariablesNames.add(reversedMap[rankList[i]]);
  }
  return orderedVariablesNames;
}

/// returns a map with 'min', 'max' and 'mean' overview for each
/// temporal variable
Future<Map<String, dynamic>> repositoryGetTemporalOverview(
    String datasetId) async {
  var response = await post(hostUrl + "getTemporalSummary", body: {
    "datasetId": datasetId,
  });
  Map<String, dynamic> overview = jsonDecode(response.body);
  return overview;
}
