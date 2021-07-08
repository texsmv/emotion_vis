import 'package:emotion_vis_client/controllers/series_controller.dart';
import 'package:emotion_vis_client/interfaces/visualizations/vis_settings.dart';
import 'package:emotion_vis_client/models/MTSerie.dart';
import 'package:emotion_vis_client/models/dataset_settings.dart';
import 'package:emotion_vis_client/utils.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:touchable/touchable.dart';

class StackedBarChartPainter extends CustomPainter {
  MTSerie mtSerie;
  int begin;
  int end;
  VisSettings visSettings;
  BuildContext context;
  int segmentsNumber;
  StackedBarChartPainter({
    @required this.mtSerie,
    @required this.visSettings,
    @required this.context,
    @required this.begin,
    @required this.end,
    this.segmentsNumber = 10,
  });

  // int get varLength => mtSerie.variablesLength;
  List<String> get variables => visSettings.variablesNames;
  int get varLength => visSettings.variablesNames.length;
  int get timeLength => mtSerie.timeLength;

  double _width;
  double _height;
  double _leftOffset = 80;
  double _rightOffset = 40;
  double _topOffset = 30;
  double _bottomOffset = 20;
  double _minBarWidth = 30;
  double _horizontalBarSpace;
  double get _graphicWidth => (_width - _rightOffset - _leftOffset);
  double get _graphicHeight => (_height - _topOffset - _bottomOffset);

  TouchyCanvas touchyCanvas;
  DatasetSettings get datasetSettings => visSettings.datasetSettings;

  Paint infoPaint;

  @override
  void paint(Canvas canvas, Size size) {
    touchyCanvas = TouchyCanvas(context, canvas);

    _width = size.width;
    _height = size.height;
    _horizontalBarSpace = _graphicWidth / (mtSerie.timeLength - 1);

    infoPaint = Paint()
      ..color = Color.fromARGB(255, 220, 220, 220)
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeWidth = 2;

    drawCanvasInfo(canvas);
    for (int i = 0; i < mtSerie.timeLength; i++) {
      drawBar(canvas, i);
    }
    if (!Get.find<SeriesController>().visualizeAllTime) {
      drawSelectedRange(canvas);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true;
  }

  void drawCanvasInfo(Canvas canvas) {
    for (int i = 0; i <= segmentsNumber; i++) {
      // draw horizontal lines
      double lineHeight = plotHeightFromValue(
          (visSettings.limitSize / segmentsNumber * i) +
              visSettings.lowerLimit);

      canvas.drawLine(
        Offset(_leftOffset, lineHeight),
        Offset(_leftOffset + _graphicWidth, lineHeight),
        infoPaint,
      );

      // draw lines text
      double value =
          (visSettings.limitSize / segmentsNumber * i) + visSettings.lowerLimit;
      TextSpan span = new TextSpan(
          style: new TextStyle(color: Colors.grey[800], fontSize: 12),
          text: value.toStringAsFixed(1));
      TextPainter tp = new TextPainter(
          text: span,
          textAlign: TextAlign.right,
          textDirection: TextDirection.ltr);
      tp.layout();
      tp.paint(
        canvas,
        new Offset(5, lineHeight - 7),
      );
    }
    for (int i = 0; i < mtSerie.timeLength; i++) {
      double textHeight = 14;
      double regionWidth = 40;
      touchyCanvas.drawRect(
        Offset(plotWidthFromValue(i.toDouble()) - regionWidth / 2,
                plotHeightFromValue(visSettings.upperLimit)) &
            Size(regionWidth, (_graphicHeight + textHeight)),
        Paint()..color = Colors.transparent,
        hitTestBehavior: HitTestBehavior.opaque,
        paintStyleForTouch: PaintingStyle.fill,
        // onHover: (event) {
        //   print("Hover $i");
        // },
        onTapDown: (_) {
          print("por fin");
        },
      );

      // TODO change this

      String timeLabel = datasetSettings.allLabels[i];
      TextSpan span = new TextSpan(
          style: new TextStyle(color: Colors.grey[800], fontSize: 12),
          text: timeLabel);
      TextPainter tp = new TextPainter(
          text: span,
          textAlign: TextAlign.left,
          textDirection: TextDirection.ltr);

      // canvas.translate(0, i * _horizontalValuesSpace);
      tp.layout();
      tp.paint(
          canvas,
          new Offset(plotWidthFromValue(i.toDouble()),
              plotHeightFromValue(visSettings.lowerLimit)));

      // String timeLabel = visSettings.timeLabels[i];
      // TextSpan span = new TextSpan(
      //     style: new TextStyle(color: Colors.grey[800], fontSize: 12),
      //     text: timeLabel);
      // TextPainter tp = new TextPainter(
      //     text: span,
      //     textAlign: TextAlign.left,
      //     textDirection: TextDirection.ltr);

      // // canvas.translate(0, i * _horizontalValuesSpace);
      // tp.layout();
      // tp.paint(
      //     canvas,
      //     new Offset(plotWidthFromValue(i.toDouble()),
      //         plotHeightFromValue(visSettings.lowerLimit)));
    }
  }

  void drawSelectedRange(Canvas canvas) {
    double barWidth;
    if (_horizontalBarSpace < _minBarWidth)
      barWidth = _horizontalBarSpace;
    else
      barWidth = _minBarWidth;
    double leftOffset = plotWidthFromValue(begin.toDouble());
    double rightOffset = plotWidthFromValue(end.toDouble());
    canvas.drawRect(
        Rect.fromPoints(
            Offset(leftOffset - barWidth / 2,
                plotHeightFromValue(visSettings.lowerLimit)),
            Offset(rightOffset - barWidth / 2,
                plotHeightFromValue(visSettings.upperLimit))),
        Paint()
          ..color = Colors.black.withAlpha(40)
          ..style = PaintingStyle.fill);
  }

  void drawBar(Canvas canvas, int timePos) {
    double barWidth;
    if (_horizontalBarSpace < _minBarWidth)
      barWidth = _horizontalBarSpace;
    else
      barWidth = _minBarWidth;
    double currBarHeigth = 0;
    for (var i = 0; i < varLength; i++) {
      double barHeight = rangeConverter(
        mtSerie.at(timePos, variables[i]),
        0,
        visSettings.upperLimit,
        0,
        _graphicHeight,
      );
      double leftOffset = plotWidthFromValue(timePos.toDouble());

      double barGraphicHeigth = rangeConverter(
        currBarHeigth,
        0,
        _graphicHeight,
        _topOffset,
        _height - _topOffset,
      );
      canvas.drawRect(
          Rect.fromPoints(
              Offset(leftOffset - barWidth / 2,
                  _graphicHeight - currBarHeigth + _topOffset),
              Offset(leftOffset + barWidth / 2,
                  _graphicHeight - (currBarHeigth + barHeight) + _topOffset)),
          Paint()
            ..color = visSettings.colors[variables[i]]
            ..style = PaintingStyle.fill);
      currBarHeigth = currBarHeigth + barHeight;
    }
  }

  double plotHeightFromValue(double value) {
    return _height -
        rangeConverter(
          value,
          visSettings.lowerLimit,
          visSettings.upperLimit,
          _bottomOffset,
          _bottomOffset + _graphicHeight,
        );
  }

  double plotWidthFromValue(double value) {
    return rangeConverter(value, 0, mtSerie.timeLength.toDouble() - 1,
        _leftOffset, _leftOffset + _graphicWidth);
  }
}
