import 'package:emotion_vis_client/interfaces/visualizations/temporal/temporal_tunnel/temporal_tunnel_painter.dart';
import 'package:emotion_vis_client/interfaces/visualizations/vis_settings.dart';
import 'package:emotion_vis_client/models/person_model.dart';
import 'package:flutter/material.dart';

class TemporalTunnel extends StatefulWidget {
  PersonModel personModel;
  VisSettings visSettings;
  TemporalTunnel({
    Key key,
    @required this.personModel,
    @required this.visSettings,
  }) : super(key: key);

  @override
  _TemporalTunnelState createState() => _TemporalTunnelState();
}

class _TemporalTunnelState extends State<TemporalTunnel> {
  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: LayoutBuilder(
            builder: (_, constraints) => Container(
              child: CustomPaint(
                size: Size(constraints.maxWidth, constraints.maxHeight),
                painter: TemporalTunnelPainter(
                  personModel: widget.personModel,
                  visSettings: widget.visSettings,
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }
}
