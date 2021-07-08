import 'package:emotion_vis_client/interfaces/visualizations/vis_settings.dart';
import 'package:emotion_vis_client/models/person_model.dart';
import 'package:flutter/material.dart';

import 'categorical_scatterplot_painter.dart';

class CategoricalScatterplot extends StatelessWidget {
  PersonModel personModel;
  VisSettings visSettings;
  CategoricalScatterplot({
    Key key,
    @required this.personModel,
    @required this.visSettings,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 20.0),
      child: Column(
        children: [
          Expanded(
            child: Container(
              height: double.infinity,
              width: double.infinity,
              child: CustomPaint(
                painter: CategoricalScatterplotPainter(
                  personModel: personModel,
                  visSettings: visSettings,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
