import 'dart:math';

import 'package:emotion_vis_client/interfaces/ui_utils.dart';
import 'package:flutter/material.dart';

class MultiDimensionalScatterplotPainter extends CustomPainter {
  List<List<int>> histogramA;
  List<List<int>> histogramB;
  Color colorA;
  Color colorB;
  int maxCellCountA;
  int maxCellCountB;

  MultiDimensionalScatterplotPainter({
    @required this.colorA,
    @required this.colorB,
    @required this.histogramA,
    @required this.histogramB,
    @required this.maxCellCountA,
    @required this.maxCellCountB,
  });

  int get nSideBins => histogramA.length;
  Canvas _canvas;
  double _width;
  double _height;
  double _binSize;
  Offset _center;
  double _side;

  @override
  void paint(Canvas canvas, Size size) {
    _width = size.width;
    _height = size.height;
    _side = min(size.width, size.height);
    _center = Offset(size.width / 2, size.height / 2);
    _binSize = _side / nSideBins;
    _canvas = canvas;

    drawCells();
    drawHistogramB();
    drawHistogramA();
  }

  void drawCells() {
    for (var i = 0; i < nSideBins; i++) {
      for (var j = 0; j < nSideBins; j++) {
        _canvas.drawRect(
          Rect.fromLTWH(
            i * _binSize,
            j * _binSize,
            _binSize,
            _binSize,
          ),
          Paint()
            ..color = Colors.black.withOpacity(0.1)
            ..style = PaintingStyle.stroke
            ..strokeWidth = 1,
        );
      }
    }
  }

  void drawHistogramA() {
    for (var i = 0; i < nSideBins; i++) {
      for (var j = 0; j < nSideBins; j++) {
        _canvas.drawRect(
            Rect.fromLTWH(
              i * _binSize,
              _height - (j + 1) * _binSize,
              _binSize,
              _binSize,
            ),
            Paint()
              ..color = colorA
                  .withOpacity(opacityByValue(histogramA[i][j], maxCellCountA))
            // .withOpacity(
            //   histogramA[i][j] / maxCellCountA,
            // ),
            );
      }
    }
  }

  void drawHistogramB() {
    for (var i = 0; i < nSideBins; i++) {
      for (var j = 0; j < nSideBins; j++) {
        _canvas.drawRect(
            Rect.fromLTWH(
              i * _binSize,
              _height - (j + 1) * _binSize,
              _binSize,
              _binSize,
            ),
            Paint()
              ..color = colorB
                  .withOpacity(opacityByValue(histogramB[i][j], maxCellCountB))
            // .withOpacity(
            //   histogramB[i][j] / (maxCellCountB),
            // ),
            );
      }
    }
  }

  double opacityByValue(int histogramVal, int maxCellCount) {
    if (histogramVal == 0) return 0;
    return uiUtilRangeConverter(
        histogramVal.toDouble(), 1, maxCellCount.toDouble(), 0.1, 1.0);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true;
  }
}
