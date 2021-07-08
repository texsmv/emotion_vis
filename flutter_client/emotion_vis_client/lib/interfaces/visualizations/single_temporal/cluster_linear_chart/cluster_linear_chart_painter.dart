import 'dart:math';

import 'package:emotion_vis_client/controllers/series_controller.dart';
import 'package:emotion_vis_client/interfaces/ui_utils.dart';
import 'package:emotion_vis_client/interfaces/visualizations/vis_settings.dart';
import 'package:emotion_vis_client/models/person_model.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:touchable/touchable.dart';
import 'package:tuple/tuple.dart';

class ClusterLinearChartPainter extends CustomPainter {
  String variableName;
  List<Tuple2<double, double>> clusterAStats;
  List<Tuple2<double, double>> clusterBStats;
  Color clusterAColor;
  Color clusterBColor;
  BuildContext context;
  VisSettings visSettings;

  ClusterLinearChartPainter({
    @required this.clusterBStats,
    @required this.clusterAStats,
    @required this.clusterAColor,
    @required this.clusterBColor,
    @required this.variableName,
    @required this.context,
    @required this.visSettings,
  });

  Paint valuesPaint;
  Paint rectPaint;
  double _width;
  double _height;
  double _horizontalSpace;
  int get timeLen => clusterAStats.length;

  Path linePath;
  Canvas _canvas;
  SeriesController _seriesController = Get.find();

  @override
  void paint(Canvas canvas, Size size) {
    _canvas = canvas;

    _width = size.width;
    _height = size.height;
    canvas.clipRect(Rect.fromLTWH(0, 0, _width, _height));

    _horizontalSpace = _width / (timeLen - 1);

    valuesPaint = Paint()
      ..color = Colors.grey
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeWidth = 2;

    rectPaint = Paint()
      ..color = Colors.blue.withAlpha(120)
      ..style = PaintingStyle.fill
      ..strokeCap = StrokeCap.round
      ..strokeWidth = 2;

    plotStandartDeviationChart(clusterAStats, clusterAColor);
    plotStandartDeviationChart(clusterBStats, clusterBColor);
  }

  void plotStandartDeviationChart(
      List<Tuple2<double, double>> clusterStats, Color color) {
    final Path errorPath = Path();
    final Paint linePaint = Paint()
      ..color = color.withOpacity(1)
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke;

    errorPath.moveTo(
      0,
      emotionValue2Heigh(clusterStats[0].item1 - clusterStats[0].item2),
    );
    errorPath.lineTo(
      0,
      emotionValue2Heigh(clusterStats[0].item1 + clusterStats[0].item2),
    );

    if (timeLen <= 30) {
      for (var i = 0; i < clusterStats.length; i++) {
        _canvas.drawCircle(
          Offset(
            i * _horizontalSpace,
            emotionValue2Heigh(
              clusterStats[i].item1,
            ),
          ),
          2,
          linePaint,
        );
      }
    }

    for (var i = 0; i < clusterStats.length - 1; i++) {
      errorPath.lineTo(
        (i + 1) * _horizontalSpace,
        emotionValue2Heigh(
          clusterStats[i + 1].item1 + clusterStats[i + 1].item2,
        ),
      );
      _canvas.drawLine(
        Offset(
          i * _horizontalSpace,
          emotionValue2Heigh(clusterStats[i].item1),
        ),
        Offset((i + 1) * _horizontalSpace,
            emotionValue2Heigh(clusterStats[i + 1].item1)),
        linePaint,
      );
    }
    errorPath.lineTo(
      _width,
      emotionValue2Heigh(
        clusterStats[timeLen - 1].item1 - clusterStats[timeLen - 1].item2,
      ),
    );
    for (var i = timeLen - 1; i > 0; i--) {
      errorPath.lineTo(
        (i - 1) * _horizontalSpace,
        emotionValue2Heigh(
            clusterStats[i - 1].item1 - clusterStats[i - 1].item2),
      );
    }
    _canvas.drawPath(
      errorPath,
      Paint()
        ..color = color.withOpacity(0.3)
        ..style = PaintingStyle.fill,
    );
  }

  double emotionValue2Heigh(double value) {
    return _height -
        uiUtilRangeConverter(value, visSettings.datasetSettings.minValue,
            visSettings.datasetSettings.maxValue, 0, _height);
    // return _height - (value / visSettings.datasetSettings.maxValue * _height);
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) {
    return true;
  }
}
