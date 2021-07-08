import 'package:emotion_vis_client/interfaces/visualizations/single_temporal/tunnel_chart/tunnel_chart_painter.dart';
import 'package:emotion_vis_client/models/person_model.dart';
import 'package:flutter/material.dart';
import 'package:touchable/touchable.dart';

class TunnelChart extends StatefulWidget {
  PersonModel personModel;
  TunnelChart({Key key, @required this.personModel}) : super(key: key);

  @override
  _TunnelChartState createState() => _TunnelChartState();
}

class _TunnelChartState extends State<TunnelChart> {
  @override
  Widget build(BuildContext context) {
    return CanvasTouchDetector(
      builder: (context) => CustomPaint(
        painter: TunnelChartPainter(
          context: context,
          personModel: widget.personModel,
        ),
      ),
    );
  }
}
