import 'package:emotion_vis_client/models/person_model.dart';
import 'package:flutter/cupertino.dart';

class TunnelChartPainter extends CustomPainter {
  PersonModel personModel;
  BuildContext context;
  TunnelChartPainter({@required this.personModel, @required this.context});

  @override
  void paint(Canvas canvas, Size size) {}

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true;
  }
}
