import 'dart:math';

import 'package:flutter/material.dart';

class LinearDensityPainter extends CustomPainter {
  List<List<int>> histogramA;
  List<List<int>> histogramB;
  Color colorA;
  Color colorB;
  int maxTemporalCellCountA;
  int maxTemporalCellCountB;

  LinearDensityPainter({
    @required this.colorA,
    @required this.histogramA,
    @required this.colorB,
    @required this.histogramB,
    @required this.maxTemporalCellCountA,
    @required this.maxTemporalCellCountB,
  });

  int get nHorizontalBins => histogramA.length;
  int get nVerticalBins => histogramA[0].length;

  Canvas _canvas;
  double _width;
  double _height;
  double _horizontalBinSize;
  double _verticalBinSize;
  Offset _center;

  @override
  void paint(Canvas canvas, Size size) {
    _width = size.width;
    _height = size.height;
    _center = Offset(size.width / 2, size.height / 2);
    _horizontalBinSize = _width / nHorizontalBins;
    _verticalBinSize = _height / nVerticalBins;
    _canvas = canvas;

    drawCells();
    drawHistogramA();
    drawHistogramB();
  }

  void drawCells() {
    for (var i = 0; i < nHorizontalBins; i++) {
      for (var j = 0; j < nVerticalBins; j++) {
        _canvas.drawRect(
          Rect.fromLTWH(
            i * _horizontalBinSize,
            j * _verticalBinSize,
            _horizontalBinSize,
            _verticalBinSize,
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
    for (var i = 0; i < nHorizontalBins; i++) {
      int count = 0;
      for (var j = 0; j < nVerticalBins; j++) {
        count = count + histogramA[i][j];
      }
      for (var j = 0; j < nVerticalBins; j++) {
        _canvas.drawRect(
          Rect.fromLTWH(
            i * _horizontalBinSize,
            _height - (j + 1) * _verticalBinSize,
            _horizontalBinSize,
            _verticalBinSize,
          ),
          Paint()
            ..color = colorA.withOpacity(
              histogramA[i][j] / count,
            ),
        );
      }
    }
  }

  void drawHistogramB() {
    for (var i = 0; i < nHorizontalBins; i++) {
      int count = 0;
      for (var j = 0; j < nVerticalBins; j++) {
        count = count + histogramB[i][j];
      }
      for (var j = 0; j < nVerticalBins; j++) {
        _canvas.drawRect(
          Rect.fromLTWH(
            i * _horizontalBinSize,
            _height - (j + 1) * _verticalBinSize,
            _horizontalBinSize,
            _verticalBinSize,
          ),
          Paint()
            ..color = colorB.withOpacity(
              histogramB[i][j] / count,
            ),
        );
      }
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true;
  }
}
