import 'dart:math';

import 'package:emotion_vis_client/interfaces/ui_utils.dart';
import 'package:emotion_vis_client/interfaces/visualizations/vis_settings.dart';
import 'package:emotion_vis_client/models/person_model.dart';
import 'package:emotion_vis_client/utils.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:rainbow_color/rainbow_color.dart';

class DimensionalScatterplotPainter extends CustomPainter {
  PersonModel personModel;
  VisSettings visSettings;
  DimensionalScatterplotPainter(
      {@required this.personModel, @required this.visSettings}) {
    axisPaint = Paint()
      ..color = Colors.black
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeWidth = 2;
  }

  Offset center;
  double width;
  double height;
  double radius;

  Paint axisPaint;
  Rainbow colorInterpolator =
      Rainbow(spectrum: [Colors.blue, Colors.red], rangeStart: 0, rangeEnd: 1);

  int get varLength => visSettings.variablesNames.length;

  @override
  void paint(Canvas canvas, Size size) {
    width = size.width;
    height = size.height;
    radius = min(size.width / 2, size.height / 2);
    center = Offset(size.width / 2, size.height / 2);

    drawAxis(canvas);
    drawPoint(canvas);
    drawAxisLabels(canvas);
  }

  void drawPoint(Canvas canvas) {
    double valenceValue;
    double arousalValue;
    double dominanceValue;

    if (varLength == 3) {
      arousalValue =
          personModel.mtSerie.at(visSettings.timePoint, visSettings.arousal);
      valenceValue =
          personModel.mtSerie.at(visSettings.timePoint, visSettings.valence);
      dominanceValue =
          personModel.mtSerie.at(visSettings.timePoint, visSettings.dominance);
    } else if (varLength == 2) {
      arousalValue =
          personModel.mtSerie.at(visSettings.timePoint, visSettings.arousal);
      valenceValue =
          personModel.mtSerie.at(visSettings.timePoint, visSettings.valence);
    }

    double valenceCanvasValue = rangeConverter(valenceValue,
        visSettings.lowerLimit, visSettings.upperLimit, -radius, radius);
    double arousalCanvasValue = rangeConverter(arousalValue,
        visSettings.lowerLimit, visSettings.upperLimit, radius, -radius);

    double dominanceRadiusValue;
    if (varLength == 3)
      dominanceRadiusValue = rangeConverter(
          dominanceValue, visSettings.lowerLimit, visSettings.upperLimit, 0, 1);

    canvas.save();
    canvas.translate(center.dx, center.dy);
    if (varLength == 3)
      canvas.drawCircle(
          Offset(valenceCanvasValue, arousalCanvasValue),
          10,
          Paint()
            ..color = colorInterpolator[dominanceRadiusValue]
            ..style = PaintingStyle.fill
            ..strokeCap = StrokeCap.round
            ..strokeWidth = 2);
    canvas.drawCircle(
        Offset(valenceCanvasValue, arousalCanvasValue), 10, axisPaint);
    canvas.restore();
  }

  void drawAxis(Canvas canvas) {
    canvas.drawLine(
        center + Offset(-radius, 0), center + Offset(radius, 0), axisPaint);

    canvas.drawLine(
        center + Offset(0, -radius), center + Offset(0, radius), axisPaint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true;
  }

  void drawAxisLabels(Canvas canvas) {
    uiUtilCanvasDrawText(
      visSettings.valence,
      canvas,
      center + Offset(radius - 40, -19),
      TextStyle(
        color: visSettings.colors[visSettings.valence],
      ),
    );

    uiUtilCanvasDrawText(
      visSettings.datasetSettings.maxValues[visSettings.valence].toString(),
      canvas,
      center + Offset(radius - 15, 0),
      TextStyle(
        color: Colors.black,
      ),
    );

    uiUtilCanvasDrawText(
      visSettings.datasetSettings.minValues[visSettings.valence].toString(),
      canvas,
      center + Offset(-radius, 0),
      TextStyle(
        color: Colors.black,
      ),
    );

    uiUtilCanvasDrawText(
      visSettings.arousal,
      canvas,
      center + Offset(0, -radius - 16),
      TextStyle(
        color: visSettings.colors[visSettings.arousal],
      ),
    );

    uiUtilCanvasDrawText(
      visSettings.datasetSettings.minValues[visSettings.arousal].toString(),
      canvas,
      center + Offset(4, radius - 15),
      TextStyle(
        color: Colors.black,
      ),
    );
    uiUtilCanvasDrawText(
      visSettings.datasetSettings.maxValues[visSettings.arousal].toString(),
      canvas,
      center + Offset(4, -radius),
      TextStyle(
        color: Colors.black,
      ),
    );
  }
}
