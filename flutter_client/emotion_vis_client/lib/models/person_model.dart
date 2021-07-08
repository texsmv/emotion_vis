import 'package:emotion_vis_client/controllers/series_controller.dart';
import 'package:emotion_vis_client/models/cluster_model.dart';
import 'package:emotion_vis_client/repositories/series_repository.dart';
import 'package:get/get.dart';
import 'dart:convert';

import 'MTSerie.dart';
import 'TSerie.dart';

class PersonModel {
  String id = "";
  MTSerie mtSerie = MTSerie(timeSeries: {}, dateTimes: []);
  double x = 0;
  double y = 0;
  Map<String, dynamic> metadata = {};
  List<String> numericalLabels = [];
  List<double> numericalValues = [];
  List<String> categoricalLabels = [];
  List<String> categoricalValues = [];
  Map<String, String> categoricalData;
  bool get isDataLoaded {
    if (mtSerie == null) return false;
    if (mtSerie.timeSeries.isEmpty) {
      return false;
    }
    return true;
  }

  RxString _clusterId = RxString();
  String get clusterId => _clusterId.value;
  set clusterId(String value) => _clusterId.value = value;

  PersonModel({this.id = "", this.mtSerie});

  PersonModel.fromMap({Map<dynamic, dynamic> map = const {}, this.id = ""}) {
    this.id = id;
    if (map != null && map != {}) {
      Map<String, TSerie> values = {};
      List<String> dimensions = List.from(map.keys.toList());
      for (int i = 0; i < dimensions.length; i++) {
        List<double> doubleList = map[dimensions[i]].cast<double>();
        values[dimensions[i]] = TSerie(values: doubleList);
      }
      // todo check this datetimes parameter
      this.mtSerie = MTSerie(timeSeries: values, dateTimes: []);
    }
  }

  ClusterModel get cluster {
    if (clusterId == null) return null;
    return Get.find<SeriesController>().clusters[clusterId];
  }

  bool identifiersContain(String searchText) {
    List<String> identifiersLabels = metadata.keys.toList();
    for (var i = 0; i < identifiersLabels.length; i++) {
      if (metadata[identifiersLabels[i]]
          .toLowerCase()
          .contains(searchText.toLowerCase())) {
        return true;
      }
    }
    return false;
  }

  Future<void> loadEmotions() async {
    SeriesController seriesController = Get.find();
    Map<String, dynamic> queryMap = await repositoryGetMTSeries(
      seriesController.selectedDatasetId,
      seriesController.datasetSettings.begin,
      seriesController.datasetSettings.end,
      [id],
      getEmotions: true,
    );

    final Map map = queryMap[id]["temporalVariables"];
    final Map<String, TSerie> values = {};
    final List<String> dimensions = List.from(map.keys.toList());
    for (int i = 0; i < dimensions.length; i++) {
      final List<double> doubleList = map[dimensions[i]].cast<double>();
      values[dimensions[i]] = TSerie(values: doubleList);
    }
    mtSerie = MTSerie(timeSeries: values, dateTimes: []);

    metadata = queryMap[id]['metadata'];
    categoricalValues = queryMap[id]['categoricalFeatures'].cast<String>();
    categoricalLabels = queryMap[id]['categoricalLabels'].cast<String>();
    numericalValues = queryMap[id]['numericalFeatures'].cast<double>();
    numericalLabels = queryMap[id]['numericalLabels'].cast<String>();
  }
}
