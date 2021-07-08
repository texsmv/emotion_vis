import 'dart:math';

import 'package:emotion_vis_client/controllers/series_controller.dart';
import 'package:emotion_vis_client/interfaces/modules/projection_view/projection_view_ui_controller.dart';
import 'package:emotion_vis_client/models/dataset_settings.dart';
import 'package:emotion_vis_client/models/visualization_levels.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:loader_overlay/loader_overlay.dart';
import 'package:intl/intl.dart' as intl;
import 'package:random_color/random_color.dart';
import 'package:tuple/tuple.dart';

extension DateOnlyCompare on DateTime {
  bool isSameDate(DateTime other) {
    return year == other.year && month == other.month && day == other.day;
  }

  bool isSameMinute(DateTime other) {
    return year == other.year &&
        month == other.month &&
        day == other.day &&
        hour == other.hour &&
        minute == other.minute;
  }

  String toDateString() {
    return intl.DateFormat.yMMMMd().format(this);
  }

  String toTimeString() {
    return intl.DateFormat('hh:mm aaa').format(this);
  }
}

String uiUtilDateTimeToStr(DateTime date, DateStrFormat format, int position) {
  switch (format) {
    case DateStrFormat.DAY_MONTH_YEAR:
      return intl.DateFormat.yMd().format(date);
      break;
    case DateStrFormat.DAY_MONTH_YEAR_HOUR_MIN:
      return intl.DateFormat.yMd().add_Hm().format(date);
      break;
    case DateStrFormat.HOUR_MIN:
      return intl.DateFormat.Hm().format(date);
      break;
    case DateStrFormat.DAY_MONTH:
      return intl.DateFormat.Md().format(date);
      break;
    case DateStrFormat.ORDER:
      return position.toString();
      break;
    default:
  }
}

String uiUtilDateFormatToStr(DateStrFormat format) {
  switch (format) {
    case DateStrFormat.DAY_MONTH_YEAR:
      return "d/m/y";
      break;
    case DateStrFormat.DAY_MONTH_YEAR_HOUR_MIN:
      return "d/m/y h:m";
      break;
    case DateStrFormat.HOUR_MIN:
      return "h:m";
      break;
    case DateStrFormat.ORDER:
      return "order";
      break;
    case DateStrFormat.DAY_MONTH:
      return "d/m";
      break;
    default:
  }
}

enum DateStrFormat {
  ORDER,
  DAY_MONTH_YEAR,
  DAY_MONTH_YEAR_HOUR_MIN,
  HOUR_MIN,
  DAY_MONTH,
}

Future<dynamic> uiUtilDialog(Widget content, {bool dismissible = true}) async {
  return Get.dialog(
    AlertDialog(
      contentPadding: const EdgeInsets.all(0),
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.all(
          Radius.circular(16.0),
        ),
      ),
      content: content,
    ),
    barrierDismissible: dismissible,
  );
}

void uiUtilShowMessage(String message) {
  uiUtilDialog(Text(message));
}

double uiUtilRandomDoubleInRange(Random source, num start, num end) =>
    source.nextDouble() * (end - start) + start;

double uiUtilRangeConverter(double oldValue, double oldMin, double oldMax,
    double newMin, double newMax) {
  return (((oldValue - oldMin) * (newMax - newMin)) / (oldMax - oldMin)) +
      newMin;
}

double uiUtilMapMin(Map<String, double> map) {
  return map.values.map((e) => e).toList().reduce(min);
}

double uiUtilMapMax(Map<String, double> map) {
  return map.values.map((e) => e).toList().reduce(max);
}

Offset uiUtilPolarToCartesian(double angle, double r) {
  return Offset(r * cos(angle), r * sin(angle));
}

String uiUtilTemVis2Str(TemporalVisualization visualization) {
  switch (visualization) {
    case TemporalVisualization.LINEAR_CHART:
      return "Linear chart";
    case TemporalVisualization.STACKED_CHART:
      return "Stack chart";
    case TemporalVisualization.POLAR_BARS:
      return "Polar bars";
    case TemporalVisualization.TAGGED_TUNNEL:
      return "Tagged tunnel";
    case TemporalVisualization.TEMPORAL_GLYPH:
      return "Glyph";
    case TemporalVisualization.TEMPORAL_TUNNEL:
      return "Tunnel";
    default:
      return "NONE";
  }
}

String uiUtilClustering2Str(ClusteringMethod method) {
  switch (method) {
    case ClusteringMethod.categorical:
      return "Categorical";
    case ClusteringMethod.dbscan:
      return "Db-scan";
    case ClusteringMethod.kmeans:
      return "K-means";
    default:
      return "None";
  }
}

String uiUtilNonTemVis2Str(NonTemporalVisualization visualization) {
  switch (visualization) {
    case NonTemporalVisualization.CATEGORICAL_SCATTERPLOT:
      return "Scatterplot";
    case NonTemporalVisualization.DIMENSIONAL_SCATTERPLOT:
      return "Scatterplot";
    case NonTemporalVisualization.INSTANT_GLYPH:
      return "Glyph";
    case NonTemporalVisualization.INSTANT_GLYPH_SINGLE:
      return "Glyph";
    case NonTemporalVisualization.POLAR_LINES:
      return "Polar lines";
    default:
      return "NONE";
  }
}

T uiUtilGetArgument<T>(int position, {T onNull = null}) {
  if (Get.arguments == null) return onNull;
  if (Get.arguments[position] is T) {
    return Get.arguments[position];
  } else {
    return onNull;
  }
}

Future<void> uiUtilDelayed(VoidCallback callback,
    {Duration delay = const Duration(milliseconds: 250)}) async {
  await Future.delayed(delay);
  callback();
}

void uiUtilShowLoaderOverlay() {
  FocusScope.of(Get.context).unfocus();
  Get.context.showLoaderOverlay();
}

void uiUtilHideLoaderOverlay() {
  FocusScope.of(Get.context).unfocus();
  Get.context.hideLoaderOverlay();
}

void uiUtilCanvasDrawText(
    String text, Canvas canvas, Offset position, TextStyle style) {
  TextSpan span = TextSpan(style: style, text: text);
  TextPainter tp = TextPainter(
    text: span,
    textAlign: TextAlign.left,
    textDirection: TextDirection.ltr,
  );
  tp.layout();
  tp.paint(canvas, position);
}

List<NonTemporalVisualization> uiUtilAvailableNonTemporalVisualizations(
  ModelType type,
  int numberOfDimensions,
) {
  List<NonTemporalVisualization> visualizations = [];
  if (type == ModelType.DISCRETE && numberOfDimensions == 2) {
    visualizations.add(NonTemporalVisualization.CATEGORICAL_SCATTERPLOT);
  }
  if (type == ModelType.DISCRETE && numberOfDimensions > 2) {
    visualizations.add(NonTemporalVisualization.POLAR_LINES);
  }
  if (type == ModelType.DIMENSIONAL &&
      numberOfDimensions >= 2 &&
      numberOfDimensions <= 3) {
    visualizations.add(NonTemporalVisualization.INSTANT_GLYPH);
  }
  if (type == ModelType.DISCRETE && numberOfDimensions == 1) {
    visualizations.add(NonTemporalVisualization.INSTANT_GLYPH_SINGLE);
  }
  if (type == ModelType.DIMENSIONAL &&
      (numberOfDimensions == 2 || numberOfDimensions == 3)) {
    visualizations.add(NonTemporalVisualization.DIMENSIONAL_SCATTERPLOT);
  }

  return visualizations;
}

List<TemporalVisualization> uiUtilAvailableTemporalVisualizations(
  bool isTagged,
  ModelType type,
  int numberOfDimensions,
) {
  List<TemporalVisualization> visualizations = [];
  // if (type == ModelType.DISCRETE && numberOfDimensions >= 1) {
  if (type == ModelType.DISCRETE || type == ModelType.DIMENSIONAL) {
    visualizations.add(TemporalVisualization.LINEAR_CHART);
  }
  if (numberOfDimensions >= 1) {
    visualizations.add(TemporalVisualization.STACKED_CHART);
  }
  if (type == ModelType.DISCRETE && numberOfDimensions >= 2 && isTagged) {
    visualizations.add(TemporalVisualization.POLAR_BARS);
  }
  if (type == ModelType.DIMENSIONAL && numberOfDimensions >= 2) {
    visualizations.add(TemporalVisualization.TEMPORAL_GLYPH);
  }
  if (type == ModelType.DISCRETE && numberOfDimensions >= 2) {
    visualizations.add(TemporalVisualization.TEMPORAL_GLYPH);
  }
  if (type == ModelType.DISCRETE && numberOfDimensions >= 2) {
    visualizations.add(TemporalVisualization.TEMPORAL_TUNNEL);
  }
  if (type == ModelType.DISCRETE && numberOfDimensions >= 2) {
    visualizations.add(TemporalVisualization.TAGGED_TUNNEL);
  }
  return visualizations;
}

String uiUtilModelTypeToStr(ModelType modelType) {
  if (modelType == ModelType.DIMENSIONAL) {
    return "dimensional";
  } else {
    return "categorical";
  }
}

String uiUtilOverviewTypeToStr(OverviewType overviewType) {
  switch (overviewType) {
    case OverviewType.MEAN:
      return "mean";
      break;
    case OverviewType.MAX:
      return "max";
      break;
    case OverviewType.MIN:
      return "min";
      break;
    default:
  }
  if (overviewType == ModelType.DIMENSIONAL) {
    return "dimensional";
  } else {
    return "categorical";
  }
}

List<Color> EMOTION_COLORS = [
  Color.fromARGB(255, 248, 15, 25),
  Color.fromARGB(255, 5, 114, 205),
  Color.fromARGB(255, 243, 208, 39),
  Color.fromARGB(255, 35, 191, 12),
  Color.fromARGB(255, 255, 128, 0),
  Color.fromARGB(255, 0, 190, 185),
  Color.fromARGB(255, 249, 70, 140),
  Color.fromARGB(255, 60, 60, 60),
  Color.fromARGB(255, 134, 62, 133),
];

List<Color> CLUSTERING_COLORS = [
  Color.fromARGB(255, 0, 133, 248),
  Color.fromARGB(255, 248, 115, 0),
  Color.fromARGB(255, 0, 136, 39),
  Color.fromARGB(255, 136, 0, 97),
  Color.fromARGB(255, 243, 208, 39),
  Color.fromARGB(255, 198, 41, 0),
  Color.fromARGB(255, 146, 202, 2),
  Color.fromARGB(255, 2, 202, 158),
];

Color uiUtilGetEmotionColor(int index) {
  if (index < EMOTION_COLORS.length) {
    return EMOTION_COLORS[index];
  }
  return RandomColor().randomColor();
}

Color uiUtilGetClusteringColor(int index) {
  if (index < CLUSTERING_COLORS.length) {
    return CLUSTERING_COLORS[index];
  }
  return RandomColor().randomColor();
}

String uiUtilDistanceToStr(int distance) {
  switch (distance) {
    case 0:
      return "Euclidean";
      break;
    case 1:
      return "DTW";
      break;
    default:
      return null;
  }
}

String uiUtilProjectionToStr(int projection) {
  switch (projection) {
    case 0:
      return "MDS";
      break;
    case 1:
      return "Isomap";
      break;
    case 2:
      return "Umap";
      break;
    case 3:
      return "T-SNE";
      break;
    default:
      return null;
  }
}

String uiUtilProjectionParamStr(int projection) {
  switch (projection) {
    case 0:
      return null;
      break;
    case 1:
      return "Neighbors";
      break;
    case 2:
      return "Neighbors";
      break;
    case 3:
      return "Perplexity";
      break;
    default:
      return null;
  }
}

Tuple2<int, int> uiUtilProjectionParamRange(int projection) {
  final int nInstances =
      Get.find<SeriesController>().datasetSettings.instanceLength;
  switch (projection) {
    case 0:
      return null;
      break;
    case 1:
      return Tuple2(2, nInstances);
      break;
    case 2:
      return Tuple2(2, nInstances);
      break;
    case 3:
      return const Tuple2(2, 50);
      break;
    default:
      return null;
  }
}

const String alphabet = 'abcdefghijklmnopqrstuvwxyz';
int uiUtilMaxAlphabetLabels() {
  return alphabet.length;
}

String uiUtilAlphabetLabel(int position) {
  return alphabet[position].toUpperCase();
}
