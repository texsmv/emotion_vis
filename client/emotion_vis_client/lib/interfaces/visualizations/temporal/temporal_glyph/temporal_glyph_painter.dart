import 'dart:math';

import 'package:emotion_vis_client/interfaces/ui_utils.dart';
import 'package:emotion_vis_client/models/dataset_settings.dart';
import 'package:emotion_vis_client/models/person_model.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';

import '../../vis_settings.dart';

class TemporalGlyphPainter extends CustomPainter {
  PersonModel personModel;
  VisSettings visSettings;

  Offset _center;
  double _width;
  double _height;
  double _radius;
  double _centerRadius = 16;
  double borderOffset = 17;
  int segmentNumber = 10;

  double angleOffset = 0 - pi / 2;
  double arcSize;

  int get timeLength => personModel.mtSerie.timeLength;
  int get varLength => visSettings.variablesNames.length;

  final _textPainter = TextPainter(textDirection: TextDirection.ltr);

  TemporalGlyphPainter({this.personModel, this.visSettings, this.borderOffset});

  @override
  void paint(Canvas canvas, Size size) {
    _width = size.width;
    _height = size.height;
    _radius = min(size.width / 2, size.height / 2);

    _center = Offset(size.width / 2, size.height / 2);
    arcSize = 2 * pi / visSettings.variablesNames.length;

    drawBackground(canvas);
    drawArcsByValues(canvas);
    drawDivisions(canvas);

    canvas.translate(size.width / 2, size.height / 2 - _radius);
    for (var i = 0; i < varLength; i++) {
      canvas.save();
      double initialAngle = 2 * pi / varLength * i + arcSize / 2;

      if (initialAngle != 0) {
        final d = 2 * _radius * sin(initialAngle / 2);
        final rotationAngle = _calculateRotationAngle(0, initialAngle);
        canvas.rotate(rotationAngle);
        canvas.translate(d, 0);
      }
      double angle = initialAngle;
      String text = visSettings.variablesNames[i];

      for (int i = 0; i < text.length; i++) {
        angle = _drawLetter(canvas, text[i], angle);
      }
      canvas.restore();
    }
  }

  void drawArcsByValues(Canvas canvas) {
    for (var i = 0; i < visSettings.variablesNames.length; i++) {
      String dimensionName = visSettings.variablesNames[i];
      double segmentArcSize = arcSize / timeLength;
      for (var j = 0; j < timeLength; j++) {
        double currRadius =
            radiusByValue(personModel.mtSerie.at(j, dimensionName));
        canvas.drawArc(
          Rect.fromCenter(
            center: _center,
            width: currRadius * 2,
            height: currRadius * 2,
          ),
          angleOffset + (arcSize * i) + segmentArcSize * j,
          segmentArcSize,
          true,
          Paint()..color = visSettings.colors[dimensionName],
        );
      }
    }
  }

  void drawDivisions(Canvas canvas) {
    Paint divisionPaint = Paint()
      ..color = Colors.white
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeWidth = 1;

    double divisionSize = 2 * pi / (varLength * timeLength);
    for (var i = 0; i < timeLength * varLength; i++) {
      canvas.drawArc(
        Rect.fromCenter(
            center: _center, width: _radius * 2, height: _radius * 2),
        angleOffset + i * divisionSize,
        divisionSize,
        true,
        divisionPaint,
      );
    }
    for (var i = 0; i < segmentNumber; i++) {
      canvas.drawCircle(
        _center,
        _centerRadius + (_radius - _centerRadius) * i / segmentNumber,
        divisionPaint,
      );
    }

    for (var i = 0; i < varLength; i++) {
      if (visSettings.timePoint != null) {
        canvas.drawArc(
          Rect.fromCenter(
              center: _center, width: _radius * 2, height: _radius * 2),
          arcSize * i + angleOffset + visSettings.timePoint * divisionSize,
          divisionSize,
          true,
          Paint()
            ..color = Colors.black.withAlpha(80)
            ..style = PaintingStyle.fill,
        );
      }
    }
  }

  void drawBackground(Canvas canvas) {
    for (var i = 0; i < visSettings.variablesNames.length; i++) {
      canvas.drawArc(
        Rect.fromCenter(
          center: _center,
          width: (_radius + borderOffset) * 2,
          height: (_radius + borderOffset) * 2,
        ),
        angleOffset + (arcSize * i),
        arcSize,
        true,
        Paint()..color = visSettings.colors[visSettings.variablesNames[i]],
      );
    }
    canvas.drawCircle(
      _center,
      _radius,
      Paint()..color = Colors.white,
    );
    for (var i = 0; i < visSettings.variablesNames.length; i++) {
      canvas.drawArc(
        Rect.fromCenter(
          center: _center,
          width: _radius * 2,
          height: _radius * 2,
        ),
        angleOffset + (arcSize * i),
        arcSize,
        true,
        Paint()
          ..color = visSettings.colors[visSettings.variablesNames[i]]
              .withOpacity(0.25),
      );
    }
    canvas.drawCircle(
      _center,
      _centerRadius,
      Paint()..color = Colors.white,
    );
  }

  double radiusByValue(double oldValue) {
    return uiUtilRangeConverter(oldValue, visSettings.lowerLimit,
        visSettings.upperLimit, _centerRadius, _radius);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true;
  }

  double _drawLetter(Canvas canvas, String letter, double prevAngle) {
    _textPainter.text = TextSpan(
        text: letter, style: TextStyle(fontSize: 14, color: Colors.white));
    _textPainter.layout(
      minWidth: 0,
      maxWidth: double.maxFinite,
    );

    final double d = _textPainter.width;
    final double alpha = 2 * asin(d / (2 * _radius));

    final newAngle = _calculateRotationAngle(prevAngle, alpha);
    canvas.rotate(newAngle);

    _textPainter.paint(canvas, Offset(0, -_textPainter.height));
    canvas.translate(d, 0);

    return alpha;
  }

  double _calculateRotationAngle(double prevAngle, double alpha) =>
      (alpha + prevAngle) / 2;
}
