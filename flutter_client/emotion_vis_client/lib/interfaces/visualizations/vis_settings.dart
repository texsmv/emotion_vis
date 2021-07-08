import 'dart:math';
import 'package:emotion_vis_client/controllers/series_controller.dart';
import 'package:emotion_vis_client/interfaces/ui_utils.dart';
import 'package:emotion_vis_client/models/dataset_settings.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:intl/intl.dart';

class VisSettings {
  Map<String, Color> colors;
  SeriesController _seriesController = Get.find();
  Map<String, double> upperLimits;
  Map<String, double> lowerLimits;
  int timePoint;
  double lowerLimit;
  double upperLimit;
  List<String> get timeLabels => datasetSettings.labels;
  List<String> get variablesNames => datasetSettings.variablesNames;

  // double get upperLimit {
  //   return upperLimits.values.map((e) => e).toList().reduce(max);
  // }

  // double get lowerLimit {
  //   return lowerLimits.values.map((e) => e).toList().reduce(min);
  // }

  String timeStr(int timePoint) {
    if (datasetSettings.isDated) {
      DateTime date = DateTime.parse(timeLabels[timePoint]);
      return date.toDateString() + " " + date.toTimeString();
    } else
      return timeLabels[timePoint];
  }

  DateTime get firstDate => DateTime.parse(timeLabels[0]);
  DateTime get lastDate => DateTime.parse(timeLabels[timeLabels.length - 1]);

  DatasetSettings get datasetSettings => _seriesController.datasetSettings;
  ModelType get modelType => datasetSettings.modelType;
  String get valence => datasetSettings.valence;
  String get dominance => datasetSettings.dominance;
  String get arousal => datasetSettings.arousal;
  double get limitSize => upperLimit - lowerLimit;

  VisSettings({
    this.colors,
    // this.variablesNames,
    this.lowerLimit,
    this.upperLimit,
    this.lowerLimits,
    this.upperLimits,
    this.timePoint,
    // this.timeLabels,
  });
}
