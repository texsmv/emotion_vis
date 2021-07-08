import 'package:emotion_vis_client/interfaces/visualizations/vis_settings.dart';
import 'package:emotion_vis_client/models/person_model.dart';
import 'package:flutter/material.dart';

import 'dimensional_scatterplot_painter.dart';

class DimensionalScatterplot extends StatelessWidget {
  PersonModel personModel;
  VisSettings visSettings;
  DimensionalScatterplot({
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
                painter: DimensionalScatterplotPainter(
                  personModel: personModel,
                  visSettings: visSettings,
                ),
              ),
            ),
          ),
          visSettings.variablesNames.length == 3
              ? Padding(
                  padding: EdgeInsets.symmetric(horizontal: 20, vertical: 5),
                  child: Column(
                    children: [
                      Container(
                        height: 8.0,
                        decoration: BoxDecoration(
                          gradient: LinearGradient(
                            colors: [Colors.blue, Colors.red],
                            begin: Alignment.centerLeft,
                            end: Alignment.centerRight,
                          ),
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Text(visSettings
                              .datasetSettings.minValues[visSettings.dominance]
                              .toString()),
                          Text(visSettings.dominance),
                          Text(visSettings
                              .datasetSettings.maxValues[visSettings.dominance]
                              .toString()),
                        ],
                      )
                    ],
                  ),
                )
              : SizedBox(),
        ],
      ),
    );
  }
}
