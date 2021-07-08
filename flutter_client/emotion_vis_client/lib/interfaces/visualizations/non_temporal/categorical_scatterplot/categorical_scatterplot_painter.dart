import 'dart:math';

import 'package:emotion_vis_client/interfaces/ui_utils.dart';
import 'package:emotion_vis_client/interfaces/visualizations/vis_settings.dart';
import 'package:emotion_vis_client/models/person_model.dart';
import 'package:emotion_vis_client/utils.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:rainbow_color/rainbow_color.dart';

class CategoricalScatterplotPainter extends CustomPainter {
  PersonModel personModel;
  VisSettings visSettings;
  CategoricalScatterplotPainter(
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

    valenceValue = personModel.mtSerie
        .at(visSettings.timePoint, visSettings.variablesNames[0]);
    arousalValue = personModel.mtSerie
        .at(visSettings.timePoint, visSettings.variablesNames[1]);

    double valenceCanvasValue = rangeConverter(valenceValue,
        visSettings.lowerLimit, visSettings.upperLimit, -radius, radius);
    double arousalCanvasValue = rangeConverter(arousalValue,
        visSettings.lowerLimit, visSettings.upperLimit, radius, -radius);

    canvas.save();
    canvas.translate(center.dx, center.dy);
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
      visSettings.variablesNames[0],
      canvas,
      center + Offset(radius - 40, -19),
      TextStyle(
        color: visSettings.colors[visSettings.variablesNames[0]],
      ),
    );

    uiUtilCanvasDrawText(
      visSettings.datasetSettings.maxValues[visSettings.variablesNames[0]]
          .toString(),
      canvas,
      center + Offset(radius - 15, 0),
      TextStyle(
        color: Colors.black,
      ),
    );

    uiUtilCanvasDrawText(
      visSettings.datasetSettings.minValues[visSettings.variablesNames[0]]
          .toString(),
      canvas,
      center + Offset(-radius, 0),
      TextStyle(
        color: Colors.black,
      ),
    );

    uiUtilCanvasDrawText(
      visSettings.variablesNames[1],
      canvas,
      center + Offset(0, -radius - 16),
      TextStyle(
        color: visSettings.colors[visSettings.variablesNames[1]],
      ),
    );

    uiUtilCanvasDrawText(
      visSettings.datasetSettings.minValues[visSettings.variablesNames[1]]
          .toString(),
      canvas,
      center + Offset(4, radius - 15),
      TextStyle(
        color: Colors.black,
      ),
    );
    uiUtilCanvasDrawText(
      visSettings.datasetSettings.maxValues[visSettings.variablesNames[1]]
          .toString(),
      canvas,
      center + Offset(4, -radius),
      TextStyle(
        color: Colors.black,
      ),
    );
  }
}
