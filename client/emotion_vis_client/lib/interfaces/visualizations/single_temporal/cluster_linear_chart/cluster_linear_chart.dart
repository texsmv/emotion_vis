import 'dart:math';

import 'package:auto_size_text/auto_size_text.dart';
import 'package:emotion_vis_client/interfaces/visualizations/vis_settings.dart';
import 'package:emotion_vis_client/models/dataset_settings.dart';
import 'package:emotion_vis_client/models/person_model.dart';
import 'package:flutter/material.dart';
import 'package:tuple/tuple.dart';

import 'cluster_linear_chart_painter.dart';

class ClusterLinearChart extends StatefulWidget {
  List<double> clusterAmeans;
  List<double> clusterBmeans;
  List<double> clusterAstd;
  List<double> clusterBstd;
  Color blueClusterColor;
  Color redClusterColor;
  String variableName;
  VisSettings visSettings;
  ClusterLinearChart({
    Key key,
    @required this.blueClusterColor,
    @required this.redClusterColor,
    @required this.variableName,
    @required this.visSettings,
    @required this.clusterAmeans,
    @required this.clusterAstd,
    @required this.clusterBmeans,
    @required this.clusterBstd,
  }) : super(key: key);

  @override
  _ClusterLinearChartState createState() => _ClusterLinearChartState();
}

class _ClusterLinearChartState extends State<ClusterLinearChart> {
  // int get timeLen => widget.clusterA.first.mtSerie.timeLength;
  int get timeLen => widget.clusterAmeans.length;
  DatasetSettings get datasetSettings => widget.visSettings.datasetSettings;
  List<Tuple2<double, double>> clusterAStats;
  List<Tuple2<double, double>> clusterBStats;
  final int n_y_divisions = 5;
  final int n_x_divisions = 10;

  @override
  void initState() {
    clusterAStats = List.generate(
        timeLen, (i) => Tuple2(widget.clusterAmeans[i], widget.clusterAstd[i]));
    clusterBStats = List.generate(
        timeLen, (i) => Tuple2(widget.clusterBmeans[i], widget.clusterBstd[i]));
    super.initState();
  }

  @override
  void didUpdateWidget(covariant ClusterLinearChart oldWidget) {
    clusterAStats = List.generate(
        timeLen, (i) => Tuple2(widget.clusterAmeans[i], widget.clusterAstd[i]));
    clusterBStats = List.generate(
        timeLen, (i) => Tuple2(widget.clusterBmeans[i], widget.clusterBstd[i]));
    super.didUpdateWidget(oldWidget);
  }

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 18),
            child: LayoutBuilder(
              builder: (context, constraints) => Container(
                width: double.infinity,
                height: double.infinity,
                child: Stack(
                  clipBehavior: Clip.none,
                  children: [
                    Positioned.fill(
                      child: Align(
                        alignment: Alignment.center,
                        child: Container(
                          child: CustomPaint(
                            size: Size(
                                constraints.maxWidth, constraints.maxHeight),
                            painter: ClusterLinearChartPainter(
                              visSettings: widget.visSettings,
                              clusterAColor: widget.blueClusterColor,
                              clusterBColor: widget.redClusterColor,
                              context: context,
                              clusterAStats: clusterAStats,
                              clusterBStats: clusterBStats,
                              variableName: widget.variableName,
                            ),
                          ),
                        ),
                      ),
                    ),
                  ]..addAll(_getAxis(constraints)),
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }

  List<Widget> _getAxis(BoxConstraints constraints) {
    final double leftStart = 0;
    final List<Widget> children = [];

    // plot left line
    children.add(Positioned(
      left: leftStart,
      child: Container(
        height: constraints.maxHeight,
        width: 2,
        color: Colors.black,
      ),
    ));

    // plot bottom line
    children.add(Positioned(
      left: leftStart,
      bottom: 0,
      child: Container(
        width: constraints.maxWidth,
        height: 2,
        color: Colors.black,
      ),
    ));

    // plot y axis line divisions
    for (var i = 0; i <= n_y_divisions; i++) {
      children.add(Positioned(
        left: leftStart,
        bottom: i * constraints.maxHeight / n_y_divisions,
        child: Container(
          width: 7,
          height: 2,
          color: Colors.black,
        ),
      ));
    }

    // plot x axis line divisions
    for (var i = 0; i <= n_x_divisions; i++) {
      children.add(Positioned(
        bottom: 0,
        left: leftStart + i * constraints.maxWidth / n_x_divisions,
        child: Container(
          height: 7,
          width: 2,
          color: Colors.black,
        ),
      ));
    }

    for (var i = 0; i <= n_y_divisions; i++) {
      children.add(
        Positioned(
          left: leftStart - 25,
          bottom: i * constraints.maxHeight / n_y_divisions - 7,
          child: Text(
            (i *
                        (datasetSettings.maxValue - datasetSettings.minValue) /
                        n_y_divisions +
                    datasetSettings.minValue)
                .toStringAsFixed(1),
            style: const TextStyle(
              fontSize: 11,
            ),
          ),
        ),
      );
    }
    for (var i = 0; i <= n_x_divisions; i++) {
      children.add(
        Positioned(
          bottom: -30,
          left: leftStart - 7 + i * constraints.maxWidth / n_x_divisions,
          child: RotatedBox(
            quarterTurns: 1,
            child: Container(
              width: 30,
              child: AutoSizeText(
                datasetSettings
                    .labels[i * ((timeLen - 1) / n_x_divisions).floor()],
                maxLines: 1,
                minFontSize: 8,
                textAlign: TextAlign.start,
                style: const TextStyle(
                  fontSize: 11,
                ),
              ),
            ),
          ),
        ),
      );
    }
    return children;
  }
}
