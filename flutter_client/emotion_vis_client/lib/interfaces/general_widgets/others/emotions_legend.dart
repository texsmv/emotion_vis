import 'dart:math';

import 'package:emotion_vis_client/controllers/series_controller.dart';
import 'package:emotion_vis_client/interfaces/constants/colors.dart';
import 'package:emotion_vis_client/interfaces/general_widgets/containers/pcard.dart';
import 'package:emotion_vis_client/interfaces/ui_utils.dart';
import 'package:emotion_vis_client/models/dataset_settings.dart';
import 'package:emotion_vis_client/models/visualization_levels.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';

class EmotionsLegend extends StatelessWidget {
  final TemporalVisualization temporalVisualization;
  EmotionsLegend({Key key, @required this.temporalVisualization})
      : super(key: key);

  SeriesController seriesController = Get.find();
  DatasetSettings get datasetSettings => seriesController.datasetSettings;

  @override
  Widget build(BuildContext context) {
    Widget content;
    switch (temporalVisualization) {
      case TemporalVisualization.TEMPORAL_TUNNEL:
        content = Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Expanded(child: _emotionsLegend()),
            Row(
              children: [
                Text(datasetSettings.minValues.values
                    .map((e) => e)
                    .toList()
                    .reduce(min)
                    .toString()),
                SizedBox(width: 5),
                Container(
                  height: 20,
                  width: 200,
                  decoration: BoxDecoration(
                      gradient: LinearGradient(
                        begin: Alignment.centerLeft,
                        end: Alignment.centerRight,
                        colors: [Colors.white, Colors.blue],
                      ),
                      border: Border.all()),
                ),
                SizedBox(width: 5),
                Text(datasetSettings.maxValues.values
                    .map((e) => e)
                    .toList()
                    .reduce(max)
                    .toString()),
              ],
            ),
          ],
        );
        break;
      default:
        content = ListView.separated(
          separatorBuilder: (context, index) => SizedBox(width: 20),
          scrollDirection: Axis.horizontal,
          itemCount: datasetSettings.variablesNames.length,
          itemBuilder: (context, index) {
            String variable = datasetSettings.variablesNames[index];
            Color color = datasetSettings.variablesColors[variable];
            return Container(
              height: 40,
              child: Row(
                children: [
                  Text(variable + ":"),
                  SizedBox(width: 10),
                  Container(
                    width: 20,
                    height: 20,
                    color: color,
                  )
                ],
              ),
            );
          },
        );
    }
    return PCard(
      child: Container(
        height: 80,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              "Emotions",
              style: TextStyle(
                fontSize: 18,
                color: pTextColorSecondary,
                fontWeight: FontWeight.w600,
              ),
            ),
            Expanded(
              child: content,
            ),
          ],
        ),
      ),
    );
  }

  Widget _emotionsLegend() {
    return ListView.separated(
      separatorBuilder: (context, index) => SizedBox(width: 20),
      scrollDirection: Axis.horizontal,
      itemCount: datasetSettings.variablesNames.length,
      itemBuilder: (context, index) {
        String variable = datasetSettings.variablesNames[index];
        String label = uiUtilAlphabetLabel(index);
        return Container(
          height: 40,
          child: Row(
            children: [
              Text(
                label + ":",
                style: TextStyle(
                  fontSize: 15,
                  fontWeight: FontWeight.w500,
                ),
              ),
              SizedBox(width: 10),
              Text(variable),
            ],
          ),
        );
      },
    );
  }
}
