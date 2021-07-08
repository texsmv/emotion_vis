import 'package:emotion_vis_client/interfaces/visualizations/temporal/temporal_glyph/temporal_glyph_painter.dart';
import 'package:emotion_vis_client/models/person_model.dart';
import 'package:flutter/material.dart';

import '../../vis_settings.dart';

class TemporalGlyph extends StatelessWidget {
  PersonModel personModel;
  VisSettings visSettings;
  double borderOffset;
  TemporalGlyph({
    Key key,
    @required this.personModel,
    @required this.visSettings,
    this.borderOffset = 17,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: EdgeInsets.all(17),
      child: CustomPaint(
        painter: TemporalGlyphPainter(
          personModel: personModel,
          visSettings: visSettings,
          borderOffset: borderOffset,
        ),
      ),
    );
  }
}
