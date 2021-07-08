import 'package:emotion_vis_client/interfaces/visualizations/overview/stacked_bar_chart.dart/stacked_bar_chart_painter.dart';
import 'package:emotion_vis_client/interfaces/visualizations/vis_settings.dart';
import 'package:emotion_vis_client/models/MTSerie.dart';
import 'package:flutter/material.dart';
import 'package:touchable/touchable.dart';

class StackedBarChart extends StatefulWidget {
  MTSerie mtSerie;
  VisSettings visSettings;
  int begin;
  int end;
  StackedBarChart({
    Key key,
    @required this.mtSerie,
    @required this.visSettings,
    @required this.begin,
    @required this.end,
  }) : super(key: key);

  @override
  _StackedBarChartState createState() => _StackedBarChartState();
}

class _StackedBarChartState extends State<StackedBarChart> {
  @override
  Widget build(BuildContext context) {
    return CanvasTouchDetector(
      builder: (touchyContext) => CustomPaint(
        painter: StackedBarChartPainter(
          context: touchyContext,
          begin: widget.begin,
          end: widget.end,
          mtSerie: widget.mtSerie,
          visSettings: widget.visSettings,
        ),
      ),
    );
  }
}
