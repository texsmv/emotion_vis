import 'dart:math';

import 'package:emotion_vis_client/interfaces/ui_utils.dart';
import 'package:emotion_vis_client/interfaces/visualizations/comparison/linear/linear_density_painter.dart';
import 'package:emotion_vis_client/interfaces/visualizations/comparison/scatterplot/multi_dimensional_scatterplot_painter.dart';
import 'package:emotion_vis_client/interfaces/visualizations/vis_settings.dart';
import 'package:emotion_vis_client/models/dataset_settings.dart';
import 'package:emotion_vis_client/models/person_model.dart';
import 'package:flutter/material.dart';

class LinearDensity extends StatefulWidget {
  final List<PersonModel> clusterA;
  final List<PersonModel> clusterB;
  final Color colorA;
  final Color colorB;
  final String variableName;
  final VisSettings visSettings;
  final int nVerticalBin;
  const LinearDensity({
    Key key,
    @required this.clusterA,
    @required this.clusterB,
    @required this.colorA,
    @required this.colorB,
    @required this.variableName,
    @required this.visSettings,
    @required this.nVerticalBin,
  }) : super(key: key);

  @override
  _LinearDensityState createState() => _LinearDensityState();
}

class _LinearDensityState extends State<LinearDensity> {
  int get timeLen => widget.clusterA.first.mtSerie.timeLength;
  DatasetSettings get datasetSettings => widget.visSettings.datasetSettings;
  List<List<int>> histogramA;
  List<List<int>> histogramB;
  int _horizontalBins;
  int get _verticalBins => widget.nVerticalBin;
  int maxHorizontalBins = 20;

  @override
  void initState() {
    createhistograms();
    fillhistogramA();
    fillhistogramB();
    super.initState();
  }

  @override
  void didUpdateWidget(covariant LinearDensity oldWidget) {
    createhistograms();
    fillhistogramA();
    fillhistogramB();
    super.didUpdateWidget(oldWidget);
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) => Container(
        width: double.infinity,
        height: double.infinity,
        child: Stack(clipBehavior: Clip.none, children: [
          Positioned.fill(
            child: CustomPaint(
              size: Size(constraints.maxHeight, constraints.maxWidth),
              painter: LinearDensityPainter(
                histogramA: histogramA,
                histogramB: histogramB,
                colorA: widget.colorA,
                colorB: widget.colorB,
                maxTemporalCellCountA: widget.clusterA.length,
                maxTemporalCellCountB: widget.clusterB.length,
              ),
            ),
          ),
        ]
            // ..addAll(_getYAxis(constraints)),
            ),
      ),
    );
  }

  void createhistograms() {
    _horizontalBins = min(timeLen, maxHorizontalBins);
    histogramA = List.generate(
      _horizontalBins,
      (index) => List.generate(_verticalBins, (_) => 0),
    );
    histogramB = List.generate(
      _horizontalBins,
      (index) => List.generate(_verticalBins, (_) => 0),
    );
  }

  void fillhistogramA() {
    for (var i = 0; i < widget.clusterA.length; i++) {
      final PersonModel person = widget.clusterA[i];
      for (var j = 0; j < timeLen; j++) {
        final int x = uiUtilRangeConverter(
          j.toDouble(),
          0.0,
          (timeLen - 1).toDouble(),
          0.0,
          (_horizontalBins - 1).toDouble(),
        ).floor();

        final int y = uiUtilRangeConverter(
          person.mtSerie.at(j, widget.variableName),
          datasetSettings.minValue,
          datasetSettings.maxValue,
          0.0,
          (_verticalBins - 1).toDouble(),
        ).floor();

        histogramA[x][y] = histogramA[x][y] + 1;
      }
    }
  }

  void fillhistogramB() {
    for (var i = 0; i < widget.clusterB.length; i++) {
      final PersonModel person = widget.clusterB[i];
      for (var j = 0; j < timeLen; j++) {
        final int x = uiUtilRangeConverter(
          j.toDouble(),
          0.0,
          (timeLen - 1).toDouble(),
          0.0,
          (_horizontalBins - 1).toDouble(),
        ).floor();

        final int y = uiUtilRangeConverter(
          person.mtSerie.at(j, widget.variableName),
          datasetSettings.minValue,
          datasetSettings.maxValue,
          0.0,
          (_verticalBins - 1).toDouble(),
        ).floor();

        histogramB[x][y] = histogramB[x][y] + 1;
      }
    }
  }
}
