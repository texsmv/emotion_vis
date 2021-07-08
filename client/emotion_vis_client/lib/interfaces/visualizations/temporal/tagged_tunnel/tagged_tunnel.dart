import 'package:emotion_vis_client/interfaces/visualizations/temporal/tagged_tunnel/tagged_tunnel_painter.dart';
import 'package:emotion_vis_client/interfaces/visualizations/vis_settings.dart';
import 'package:emotion_vis_client/models/person_model.dart';
import 'package:flutter/material.dart';

class TaggedTunnel extends StatefulWidget {
  final PersonModel personModel;
  final VisSettings visSettings;
  const TaggedTunnel({
    Key key,
    @required this.personModel,
    @required this.visSettings,
  }) : super(key: key);

  @override
  _TaggedTunnelState createState() => _TaggedTunnelState();
}

class _TaggedTunnelState extends State<TaggedTunnel> {
  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: TaggedTunnelPainter(
        personModel: widget.personModel,
        visSettings: widget.visSettings,
      ),
    );
  }
}
