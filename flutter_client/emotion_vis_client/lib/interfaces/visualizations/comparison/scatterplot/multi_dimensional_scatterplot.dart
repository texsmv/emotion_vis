import 'dart:math';

import 'package:emotion_vis_client/interfaces/modules/projection_view/projection_view_ui_controller.dart';
import 'package:emotion_vis_client/interfaces/visualizations/comparison/scatterplot/multi_dimensional_scatterplot_painter.dart';
import 'package:emotion_vis_client/interfaces/visualizations/vis_settings.dart';
import 'package:emotion_vis_client/models/dataset_settings.dart';
import 'package:emotion_vis_client/models/person_model.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';

class MultiDimensionalScatterplot extends StatefulWidget {
  final List<PersonModel> clusterA;
  final List<PersonModel> clusterB;
  final Color colorA;
  final Color colorB;
  final VisSettings visSettings;
  final int nSideBins;
  const MultiDimensionalScatterplot({
    Key key,
    @required this.clusterA,
    @required this.clusterB,
    @required this.colorA,
    @required this.colorB,
    @required this.visSettings,
    @required this.nSideBins,
  }) : super(key: key);

  @override
  _MultiDimensionalScatterplotState createState() =>
      _MultiDimensionalScatterplotState();
}

class _MultiDimensionalScatterplotState
    extends State<MultiDimensionalScatterplot> {
  ProjectionViewUiController _viewController = Get.find();
  int get timeLen => widget.clusterA.first.mtSerie.timeLength;
  DatasetSettings get datasetSettings => widget.visSettings.datasetSettings;
  List<List<int>> get histogramA => _viewController.histogramA;
  List<List<int>> get histogramB => _viewController.histogramB;
  int get maxCellCountA => _viewController.histogramAmaxCount;
  int get maxCellCountB => _viewController.histogramBmaxCount;

  @override
  void initState() {
    // createHistogram();
    // fillHistograms();
    super.initState();
  }

  @override
  void didUpdateWidget(covariant MultiDimensionalScatterplot oldWidget) {
    // createHistogram();
    // fillHistograms();
    super.didUpdateWidget(oldWidget);
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) => Container(
        width: double.infinity,
        height: double.infinity,
        child: Stack(
          clipBehavior: Clip.none,
          children: [
            Positioned.fill(
              child: Align(
                alignment: Alignment.center,
                child: CustomPaint(
                  size: Size(constraints.maxHeight, constraints.maxHeight),
                  painter: MultiDimensionalScatterplotPainter(
                    histogramA: histogramA,
                    histogramB: histogramB,
                    maxCellCountA: maxCellCountA,
                    maxCellCountB: maxCellCountB,
                    colorA: widget.colorA,
                    colorB: widget.colorB,
                  ),
                ),
              ),
            ),
          ]..addAll(_getYAxis(constraints)),
        ),
      ),
    );
  }

  List<Widget> _getYAxis(BoxConstraints constraints) {
    final double leftStart = (constraints.maxWidth - constraints.maxHeight) / 2;
    final List<Widget> children = [];

    children.add(Positioned(
      left: leftStart,
      child: Container(
        height: constraints.maxHeight,
        width: 1,
        color: Colors.black,
      ),
    ));

    children.add(Positioned(
      left: leftStart,
      bottom: 0,
      child: Container(
        width: constraints.maxHeight,
        height: 1,
        color: Colors.black,
      ),
    ));
    children.add(
      Positioned(
        left: leftStart - 40,
        child: RotatedBox(
          quarterTurns: 3,
          child: Container(
            height: 13,
            // width: double.infinity,
            child: Text("Arousal"),
          ),
        ),
      ),
    );
    children.add(
      Positioned(
        left: leftStart + constraints.maxHeight - 35,
        bottom: -40,
        child: RotatedBox(
          quarterTurns: 0,
          child: Container(
            height: 13,
            // width: double.infinity,
            child: Text("Valence"),
          ),
        ),
      ),
    );

    for (var i = 0; i <= widget.nSideBins; i++) {
      children.add(Positioned(
        left: leftStart,
        bottom: i * constraints.maxHeight / widget.nSideBins,
        child: Container(
          width: 7,
          height: 1,
          color: Colors.black,
        ),
      ));
    }

    for (var i = 0; i <= widget.nSideBins; i++) {
      children.add(Positioned(
        bottom: 0,
        left: leftStart + i * constraints.maxHeight / widget.nSideBins,
        child: Container(
          height: 7,
          width: 1,
          color: Colors.black,
        ),
      ));
    }

    for (var i = 0; i <= widget.nSideBins; i++) {
      if (i % 4 == 0) {
        children.add(
          Positioned(
            left: leftStart - 22,
            bottom: i * constraints.maxHeight / widget.nSideBins - 7,
            child: Text(
              (i *
                          (datasetSettings.maxValue -
                              datasetSettings.minValue) /
                          widget.nSideBins +
                      datasetSettings.minValue)
                  .toStringAsFixed(1),
            ),
          ),
        );
      }
    }

    for (var i = 0; i <= widget.nSideBins; i++) {
      if (i % 4 == 0) {
        children.add(
          Positioned(
            bottom: -22,
            left: leftStart - 7 + i * constraints.maxHeight / widget.nSideBins,
            child: RotatedBox(
              quarterTurns: 1,
              child: Text(
                (i *
                            (datasetSettings.maxValue -
                                datasetSettings.minValue) /
                            widget.nSideBins +
                        datasetSettings.minValue)
                    .toStringAsFixed(1),
              ),
            ),
          ),
        );
      }
    }
    return children;
  }

  // void createHistogram() {
  //   maxCellCountA = 0;
  //   maxCellCountB = 0;
  //   histogramA = List.generate(
  //     widget.nSideBins,
  //     (index) => List.generate(widget.nSideBins, (_) => 0),
  //   );
  //   histogramB = List.generate(
  //     widget.nSideBins,
  //     (index) => List.generate(widget.nSideBins, (_) => 0),
  //   );
  // }

  // void fillHistograms() {
  //   for (var i = 0; i < widget.clusterA.length; i++) {
  //     final PersonModel person = widget.clusterA[i];
  //     for (var j = 0; j < timeLen; j++) {
  //       final int x = uiUtilRangeConverter(
  //         person.mtSerie.at(j, datasetSettings.valence),
  //         widget.visSettings.datasetSettings.minValues[datasetSettings.valence],
  //         widget.visSettings.datasetSettings.maxValues[datasetSettings.valence],
  //         0.0,
  //         (widget.nSideBins - 1).toDouble(),
  //       ).floor();
  //       final int y = uiUtilRangeConverter(
  //         person.mtSerie.at(j, datasetSettings.arousal),
  //         widget.visSettings.datasetSettings.minValues[datasetSettings.arousal],
  //         widget.visSettings.datasetSettings.maxValues[datasetSettings.arousal],
  //         0.0,
  //         (widget.nSideBins - 1).toDouble(),
  //       ).floor();
  //       histogramA[x][y] = histogramA[x][y] + 1;
  //       if (histogramA[x][y] > maxCellCountA) {
  //         maxCellCountA = histogramA[x][y];
  //       }
  //     }
  //   }

  //   for (var i = 0; i < widget.clusterB.length; i++) {
  //     final PersonModel person = widget.clusterB[i];
  //     for (var j = 0; j < timeLen; j++) {
  //       final int x = uiUtilRangeConverter(
  //         person.mtSerie.at(j, datasetSettings.valence),
  //         widget.visSettings.datasetSettings.minValues[datasetSettings.valence],
  //         widget.visSettings.datasetSettings.maxValues[datasetSettings.valence],
  //         0.0,
  //         (widget.nSideBins - 1).toDouble(),
  //       ).floor();
  //       final int y = uiUtilRangeConverter(
  //         person.mtSerie.at(j, datasetSettings.arousal),
  //         widget.visSettings.datasetSettings.minValues[datasetSettings.arousal],
  //         widget.visSettings.datasetSettings.maxValues[datasetSettings.arousal],
  //         0.0,
  //         (widget.nSideBins - 1).toDouble(),
  //       ).floor();
  //       histogramB[x][y] = histogramB[x][y] + 1;
  //       if (histogramB[x][y] > maxCellCountB) {
  //         maxCellCountB = histogramB[x][y];
  //       }
  //     }
  //   }
  // }
}
