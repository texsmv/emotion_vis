import 'dart:math';

import 'package:emotion_vis_client/controllers/series_controller.dart';
import 'package:emotion_vis_client/interfaces/ui_utils.dart';
import 'package:emotion_vis_client/interfaces/visualizations/vis_settings.dart';
import 'package:emotion_vis_client/models/MTSerie.dart';
import 'package:emotion_vis_client/models/dataset_settings.dart';
import 'package:emotion_vis_client/models/person_model.dart';
import 'package:emotion_vis_client/models/visualization_levels.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:rainbow_color/rainbow_color.dart';

class PolarBarsPainter extends CustomPainter {
  PersonModel personModel;
  VisSettings visSettings;

  PolarBarsPainter({
    @required this.personModel,
    @required this.visSettings,
  });

  Paint infoPaint;
  Paint rectPaint;
  Paint timePointPaint;

  double _width;
  double _height;
  Offset _center;
  double _radius;
  double _chartRadius;
  double _centerRadius;
  double _ringWidth;
  double _arcSize;
  int _nDivisions = 3;
  Canvas _canvas;
  int get timeLength => personModel.mtSerie.timeLength;

  final _textPainter = TextPainter(textDirection: TextDirection.ltr);
  List<Offset> meanPoints;
  List<String> get variables => visSettings.variablesNames;
  SeriesController _seriesController = Get.find();
  Map<String, List<double>> get meanOverview =>
      _seriesController.overview[OverviewType.MEAN];
  DatasetSettings get datasetSettings => visSettings.datasetSettings;
  MTSerie get mtserie => personModel.mtSerie;
  bool get useAllLabels =>
      mtserie.timeLength == datasetSettings.allLabels.length;
  List<String> get labels =>
      useAllLabels ? datasetSettings.allLabels : datasetSettings.labels;

  @override
  void paint(Canvas canvas, Size size) {
    _canvas = canvas;
    _width = size.width;
    _height = size.height;
    _center = Offset(size.width / 2, size.height / 2);
    _radius = min(size.width / 2, size.height / 2);
    _chartRadius = _radius * 0.8;
    _centerRadius = _radius * 0.25;

    _arcSize = 2 * pi / variables.length;

    drawBackground();
    drawSegments();

    double _textRadius = _radius * 0.8;
    canvas.translate(size.width / 2, size.height / 2 - _textRadius);
    for (var i = 0; i < timeLength; i++) {
      canvas.save();
      double initialAngle = 2 * pi / timeLength * i;

      if (initialAngle != 0) {
        final d = 2 * _textRadius * sin(initialAngle / 2);
        final rotationAngle = _calculateRotationAngle(0, initialAngle);
        canvas.rotate(rotationAngle);
        canvas.translate(d, 0);
      }
      double angle = initialAngle;
      String text = visSettings.timeLabels[i];

      for (int i = 0; i < min(text.length, 10); i++) {
        angle = _drawLetter(canvas, text[i], angle);
      }
      canvas.restore();
    }
  }

  double _drawLetter(Canvas canvas, String letter, double prevAngle) {
    _textPainter.text = TextSpan(
        text: letter, style: TextStyle(fontSize: 14, color: Colors.black));
    _textPainter.layout(
      minWidth: 0,
      maxWidth: double.maxFinite,
    );

    final double d = _textPainter.width;
    final double alpha = 2 * asin(d / (2 * _radius * 0.9));

    final newAngle = _calculateRotationAngle(prevAngle, alpha);
    canvas.rotate(newAngle);

    _textPainter.paint(canvas, Offset(0, -_textPainter.height));
    canvas.translate(d, 0);

    return alpha;
  }

  double _calculateRotationAngle(double prevAngle, double alpha) =>
      (alpha + prevAngle) / 2;

  void drawCenter() {
    _canvas.drawCircle(_center, _centerRadius, Paint()..color = Colors.grey);
  }

  void drawBackground() {
    _ringWidth = (_chartRadius) / _nDivisions;
    for (var i = 0; i < _nDivisions; i++) {
      _canvas.drawCircle(
          _center,
          _ringWidth * (i + 1),
          Paint()
            ..color = Colors.black
            ..style = PaintingStyle.stroke);
    }
    for (var i = 0; i < timeLength; i++) {
      _canvas.drawLine(
        _center,
        _center + Offset.fromDirection(pi * 2 / timeLength * i, _chartRadius),
        Paint()
          ..color = Colors.black
          ..style = PaintingStyle.stroke,
      );
    }
  }

  void drawSegments() {
    Path meanPath = Path();
    meanPoints = [];
    for (var i = 0; i < timeLength; i++) {
      drawSegment(i, pi * 2 / timeLength * i, pi * 2 / timeLength);

      drawSegmentMeans(i, pi * 2 / timeLength * i, pi * 2 / timeLength);
    }
    meanPath.moveTo(meanPoints[0].dx, meanPoints[0].dy);
    for (var i = 1; i < meanPoints.length; i++) {
      meanPath.lineTo(meanPoints[i].dx, meanPoints[i].dy);
    }
    meanPath.lineTo(meanPoints[0].dx, meanPoints[0].dy);
    _canvas.drawPath(
        meanPath,
        Paint()
          ..color = Colors.black
          ..style = PaintingStyle.stroke);
  }

  void drawSegment(int timePoint, double startAngle, double segmentSize) {
    double arcSize = segmentSize / variables.length;
    for (var i = 0; i < variables.length; i++) {
      double value = uiUtilRangeConverter(
        personModel.mtSerie.at(timePoint, variables[i]),
        visSettings.lowerLimits[variables[i]],
        visSettings.upperLimits[variables[i]],
        0,
        _chartRadius * 2,
      );
      _canvas.drawArc(
          Rect.fromCenter(center: _center, width: value, height: value),
          startAngle + arcSize * i,
          arcSize,
          true,
          Paint()..color = visSettings.colors[variables[i]]);
    }
  }

  void drawSegmentMeans(int timePoint, double startAngle, double segmentSize) {
    double arcSize = segmentSize / variables.length;
    for (var i = 0; i < variables.length; i++) {
      double mean = uiUtilRangeConverter(
        meanOverview[variables[i]][timePoint],
        visSettings.lowerLimits[variables[i]],
        visSettings.upperLimits[variables[i]],
        0,
        _chartRadius,
      );
      Offset pointPosition = _center +
          Offset.fromDirection(startAngle + arcSize * i + arcSize / 2, mean);
      meanPoints.add(pointPosition);
      // _canvas.drawCircle(pointPosition, 3, Paint()..color = Colors.black);
    }
  }

  // // for each emotion
  // void drawArcs(int position) {
  //   String emotion = variables[position];
  //   for (var i = personModel.mtSerie.timeLength - 1; i >= 0; i--) {
  //     final arcRadius = _centerRadius + _ringWidth * (i + 1);
  //     double scaledValue = uiUtilRangeConverter(
  //       personModel.mtSerie.at(i, emotion),
  //       visSettings.lowerLimits[emotion],
  //       visSettings.upperLimits[emotion],
  //       0,
  //       1,
  //     );
  //     _canvas.drawArc(
  //       Rect.fromCenter(
  //           center: _center, width: arcRadius * 2, height: arcRadius * 2),
  //       _arcSize * position,
  //       _arcSize,
  //       true,
  //       Paint()
  //         ..color = colorInterpolator[scaledValue]
  //         ..style = PaintingStyle.fill,
  //     );
  //     _canvas.drawArc(
  //       Rect.fromCenter(
  //           center: _center, width: arcRadius * 2, height: arcRadius * 2),
  //       _arcSize * position,
  //       _arcSize,
  //       true,
  //       Paint()
  //         ..color = Colors.black
  //         ..style = PaintingStyle.stroke,
  //     );
  //   }
  // }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) {
    return true;
  }
}
