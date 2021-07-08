import 'dart:math';

import 'package:emotion_vis_client/interfaces/constants/app_constants.dart';
import 'package:emotion_vis_client/interfaces/ui_utils.dart';
import 'package:emotion_vis_client/models/visualization_levels.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';

class DatasetSettings {
  // * Info
  List<String> ids = [];
  Map<String, double> minValues = {};
  Map<String, double> maxValues = {};
  List<String> variablesNames = [];
  int timeLength = 0;
  int instanceLength = 0;
  int variablesLength = 0;
  List<DateTime> dateTimes = [];
  List<String> dateLabels = []; // * only available if [isDated]
  List<String> tags = []; // * only available if [isTagged]
  List<String> _labels; // * obtained from dates or tags or generated
  List<String> get labels => _labels.sublist(begin, end);
  List<String> get allLabels => _labels;
  bool isDated = false;
  bool isTagged = false;
  List<String> downsampleRules = [];
  Map<String, List<dynamic>> categoricalData = {};
  List<String> numericalLabels = [];
  List<String> get categoricalLabels => categoricalData.keys.toList();
  List<String> identifiersLabels = [];
  String valence;
  String dominance;
  String arousal;
  ModelType modelType = ModelType.DISCRETE;

  // * Settings
  final RxBool _updating = false.obs;
  bool get updating => _updating.value;
  set updating(bool value) => _updating.value = value;
  OverviewType overviewType = OverviewType.values[0];
  DateStrFormat dateFormat = DateStrFormat.values[0];
  int distance = 0;
  int projection = 0;
  int get projectionParameter => parameters[projection];
  int getProjectionParameter(int pProjection) => parameters[pProjection];
  set projectionParameter(int value) => parameters[projection] = value;
  List<int> parameters = [null, 5, 5, 5];

  int windowLength;
  int windowPosition;
  Map<String, double> alphas = {};
  Map<String, Color> variablesColors = {};
  DatasetSettings();

  DatasetSettings.fromMap({
    Map<String, dynamic> info,
    this.windowPosition,
    this.windowLength,
  }) {
    ids = List<String>.from(info[INFO_IDS]);
    minValues = Map<String, double>.from(info[INFO_MIN_VALUES]);
    maxValues = Map<String, double>.from(info[INFO_MAX_VALUES]);
    variablesNames = List<String>.from(info[INFO_SERIES_LABELS]);
    if (info.containsKey(INFO_CATEGORICAL_LABELS)) {
      categoricalData = Map<String, List>.from(info[INFO_CATEGORICAL_LABELS]);
    }
    if (info.containsKey(INFO_NUMERICAL_LABELS)) {
      numericalLabels = List<String>.from(info[INFO_NUMERICAL_LABELS]);
    }
    identifiersLabels = List<String>.from(info[INFO_IDENTIFIERS_LABELS]);
    timeLength = info[INFO_LEN_TIME];
    instanceLength = info[INFO_LEN_INSTANCE];
    variablesLength = variablesNames.length;
    isDated = info[INFO_IS_DATED];
    if (isDated) {
      List<String> dates = List<String>.from(info[INFO_DATES]);
      dateTimes =
          List.generate(dates.length, (index) => DateTime.parse(dates[index]));
      formatDateTimes();
      downsampleRules = List<String>.from(info[INFO_DOWNSAMPLE_RULES]);
      useDatesAsLabels();
    }
    isTagged = info.containsKey(INFO_LABELS);
    if (isTagged) {
      tags = List<String>.from(info[INFO_LABELS]);
      useTagsAsLabels();
    }
    if (!isDated && !isTagged) {
      usePositionsAsLabels();
    }
    modelType = info[INFO_TYPE] == "dimensional"
        ? ModelType.DIMENSIONAL
        : ModelType.DISCRETE;
    if (modelType == ModelType.DIMENSIONAL) {
      Map<String, String> dimensionsMap =
          Map<String, String>.from(info[INFO_DIMENSIONS]);
      for (int i = 0; i < variablesLength; i++) {
        String dim = dimensionsMap[variablesNames[i]];
        if (dim == "valence") {
          valence = variablesNames[i];
        } else if (dim == "arousal") {
          arousal = variablesNames[i];
        } else if (dim == "dominance") {
          dominance = variablesNames[i];
        }
      }
    }

    // settings

    for (var i = 0; i < variablesLength; i++) {
      alphas[variablesNames[i]] = 1;
      variablesColors[variablesNames[i]] = uiUtilGetEmotionColor(i);
    }

    windowPosition = 0;
    windowLength = timeLength;
  }

  void formatDateTimes() {
    dateLabels = List.generate(dateTimes.length,
        (index) => uiUtilDateTimeToStr(dateTimes[index], dateFormat, index));
    _labels = dateLabels;
  }

  bool useDatesAsLabels() {
    if (!isDated) return false;
    _labels = dateLabels;
    return true;
  }

  bool useTagsAsLabels() {
    if (!isTagged) return false;
    _labels = tags;
    return true;
  }

  void usePositionsAsLabels() {
    _labels = List.generate(timeLength, (index) => index.toString());
  }

  // ----------------------------getters---------------------------------------
  bool get hasCategoricalMetadata => categoricalLabels.isNotEmpty;
  bool get hasNumericalMetadata => numericalLabels.isNotEmpty;
  String get firstLabel => _labels[0];

  String get lastLabel => _labels[_labels.length - 1];

  DateTime get firstDate => dateTimes[0];
  DateTime get lastDate => dateTimes[dateTimes.length - 1];

  int get begin => windowPosition;
  int get end => windowPosition + windowLength;

  set begin(int value) => windowPosition = begin;
  set end(int value) => windowLength = value - begin;

  double get maxValue {
    return maxValues.values.map((e) => e).toList().reduce(max);
  }

  double get minValue {
    return minValues.values.map((e) => e).toList().reduce(min);
  }
}

enum DimensionalDimension {
  NONE,
  AROUSAL,
  VALENCE,
  DOMINANCE,
}

enum ModelType { DIMENSIONAL, DISCRETE }
