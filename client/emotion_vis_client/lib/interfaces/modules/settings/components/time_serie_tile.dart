import 'package:emotion_vis_client/interfaces/modules/settings/settings_ui_controller.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';

class TimeSerieTile extends GetView<SettingsUiController> {
  final String varName;
  TimeSerieTile({Key key, this.varName}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 60,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Container(
            width: 120,
            child: Text(varName),
          ),
          Row(
            children: [
              Container(
                decoration: BoxDecoration(
                  border: Border.all(
                    color: Colors.white.withAlpha(0),
                  ),
                  color: controller.datasetSettings.variablesColors[varName],
                  shape: BoxShape.circle,
                ),
                height: 20,
                width: 20,
              ),
              const SizedBox(width: 10),
              IconButton(
                icon: const Icon(Icons.edit),
                onPressed: () {
                  controller.editTimeSerieItem(varName);
                },
              )
            ],
          ),
        ],
      ),
    );
  }
}
