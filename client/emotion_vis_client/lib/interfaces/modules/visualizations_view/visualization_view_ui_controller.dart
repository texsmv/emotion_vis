import 'package:emotion_vis_client/controllers/series_controller.dart';
import 'package:emotion_vis_client/interfaces/modules/home/home_ui_controller.dart';
import 'package:emotion_vis_client/models/dataset_settings.dart';
import 'package:emotion_vis_client/models/person_model.dart';
import 'package:emotion_vis_client/models/visualization_levels.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';

const String noneCluster = "None";

class VisualizationsViewUiController extends GetxController {
  final HomeUiController _homeUiController = Get.find();
  final SeriesController _seriesController = Get.find();

  bool get isDataClustered => _seriesController.clusters.isNotEmpty;
  List<PersonModel> get persons {
    List<PersonModel> filteredPersons;
    if (selectedCluster == noneCluster) {
      filteredPersons = _seriesController.persons;
    } else {
      filteredPersons = _seriesController.clusters[selectedCluster].persons;
    }
    if (searchController.text == null || searchController.text == "") {
      return filteredPersons;
    }

    final List<PersonModel> searchPersons = [];

    for (var i = 0; i < filteredPersons.length; i++) {
      if (filteredPersons[i].identifiersContain(searchController.text)) {
        searchPersons.add(filteredPersons[i]);
      }
    }

    return searchPersons;
  }

  ScrollController listScrollController = ScrollController();
  Map<String, Color> get colors =>
      _seriesController.datasetSettings.variablesColors;
  Map<String, double> get lowerLimits =>
      _seriesController.datasetSettings.minValues;
  Map<String, double> get upperLimits =>
      _seriesController.datasetSettings.maxValues;

  DatasetSettings get datasetSettings => _seriesController.datasetSettings;
  TemporalVisualization get temporalVisualization =>
      _homeUiController.temporalVisualization;
  NonTemporalVisualization get nonTemporalVisualization =>
      _homeUiController.nonTemporalVisualization;

  TextEditingController searchController;
  List<String> get clustersOptions =>
      _seriesController.clustersIds..add(noneCluster);
  String selectedCluster = noneCluster;

  @override
  void onInit() {
    searchController = TextEditingController();
    searchController.addListener(() {
      update();
    });
    super.onInit();
  }

  void selectCluster(String selection) {
    selectedCluster = selection;
    update();
  }

  void resetSearch() {
    searchController.text = "";
    update();
  }
}
