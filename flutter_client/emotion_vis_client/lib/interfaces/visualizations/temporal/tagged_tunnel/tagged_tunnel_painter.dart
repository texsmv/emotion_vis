import 'dart:math';

import 'package:emotion_vis_client/interfaces/ui_utils.dart';
import 'package:emotion_vis_client/interfaces/visualizations/vis_settings.dart';
import 'package:emotion_vis_client/models/MTSerie.dart';
import 'package:emotion_vis_client/models/dataset_settings.dart';
import 'package:emotion_vis_client/models/person_model.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:rainbow_color/rainbow_color.dart';
import 'package:touchable/touchable.dart';
import 'package:tuple/tuple.dart';

class TaggedTunnelPainter extends CustomPainter {
  PersonModel personModel;
  VisSettings visSettings;

  TaggedTunnelPainter({
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
  double _tunnelRadius;
  double _centerRadius;
  double _ringWidth;
  double _arcSize;
  Canvas _canvas;
  final _textPainter = TextPainter(textDirection: TextDirection.ltr);

  List<String> get variables => visSettings.variablesNames;
  int get timeLength => personModel.mtSerie.timeLength;
  DatasetSettings get datasetSettings => visSettings.datasetSettings;
  MTSerie get mtserie => personModel.mtSerie;
  bool useAllLabels;

  @override
  void paint(Canvas canvas, Size size) {
    useAllLabels = mtserie.timeLength == datasetSettings.allLabels.length;
    _canvas = canvas;
    _width = size.width;
    _height = size.height;
    _center = Offset(size.width / 2, size.height / 2);
    _radius = min(size.width / 2, size.height / 2);
    _tunnelRadius = _radius * 0.8;
    _centerRadius = _radius * 0.25;

    _ringWidth = (_tunnelRadius - _centerRadius) / variables.length;

    _arcSize = 2 * pi / timeLength;

    drawRings();
    drawCenter();

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
      String text = useAllLabels
          ? datasetSettings.allLabels[i]
          : datasetSettings.labels[i];

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

  void drawRings() {
    for (var i = 0; i < timeLength; i++) {
      drawArcs(i);
    }
  }

  List<String> getOrderedEmotions(int timePoint) {
    List<Tuple2<String, double>> tuples = List.generate(
        variables.length,
        (index) => Tuple2(variables[index],
            personModel.mtSerie.at(timePoint, variables[index])));
    tuples.sort((a, b) => a.item2.compareTo(b.item2));
    return List.generate(tuples.length, (index) => tuples[index].item1);
  }

  // for each time point
  void drawArcs(int timePoint) {
    List<String> orderedEmotions = getOrderedEmotions(timePoint);
    for (var i = variables.length - 1; i >= 0; i--) {
      String emotion = orderedEmotions[i];
      final arcRadius = _centerRadius + _ringWidth * (i + 1);
      double scaledValue = uiUtilRangeConverter(
        personModel.mtSerie.at(timePoint, emotion),
        visSettings.lowerLimits[emotion],
        visSettings.upperLimits[emotion],
        0,
        1,
      );
      _canvas.drawArc(
        Rect.fromCenter(
            center: _center, width: arcRadius * 2, height: arcRadius * 2),
        _arcSize * timePoint,
        _arcSize,
        true,
        Paint()
          ..color = visSettings.colors[emotion]
          ..style = PaintingStyle.fill,
      );
      _canvas.drawArc(
        Rect.fromCenter(
            center: _center, width: arcRadius * 2, height: arcRadius * 2),
        _arcSize * timePoint,
        _arcSize,
        true,
        Paint()
          ..color = Colors.black
          ..style = PaintingStyle.stroke,
      );
    }
  }

  
  @override
  bool shouldRepaint(CustomPainter oldDelegate) {
    return true;
  }
}
