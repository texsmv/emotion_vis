import 'package:emotion_vis_client/interfaces/constants/colors.dart';
import 'package:emotion_vis_client/interfaces/modules/single_person/single_person_ui_controller.dart';
import 'package:emotion_vis_client/models/dataset_settings.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';

class TimeProgress extends GetView<SinglePersonUiController> {
  const TimeProgress({Key key}) : super(key: key);

  DatasetSettings get datasetSettings => controller.datasetSettings;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Text(
          "Selected time point:",
          style: TextStyle(
            color: pColorPrimary,
            fontWeight: FontWeight.w600,
            fontSize: 16,
          ),
        ),
        Expanded(
          child: Column(
            children: [
              SliderTheme(
                data: SliderThemeData(
                  thumbColor: pColorAccent,
                  activeTrackColor: pColorGray,
                  inactiveTrackColor: pColorGray,
                ),
                child: Slider(
                  value: controller.timePoint.toDouble(),
                  min: 0,
                  max: controller.personModel != null
                      ? controller.personModel.mtSerie.timeLength.toDouble() - 1
                      : 1,
                  onChanged: (value) {
                    controller.timePoint = value.toInt();
                    controller.update();
                  },
                ),
              ),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 25),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(datasetSettings.firstLabel),
                    Text(datasetSettings.lastLabel),
                  ],
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}
