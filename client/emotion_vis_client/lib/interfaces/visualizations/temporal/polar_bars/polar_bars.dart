import 'package:emotion_vis_client/interfaces/visualizations/temporal/polar_bars/polar_bars_painter.dart';
import 'package:emotion_vis_client/interfaces/visualizations/vis_settings.dart';
import 'package:emotion_vis_client/models/person_model.dart';
import 'package:flutter/material.dart';

class PolarBars extends StatelessWidget {
  final PersonModel personModel;
  final VisSettings visSettings;
  const PolarBars({
    Key key,
    @required this.personModel,
    @required this.visSettings,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: PolarBarsPainter(
        personModel: personModel,
        visSettings: visSettings,
      ),
    );
  }
}
