import 'package:emotion_vis_client/interfaces/constants/colors.dart';
import 'package:emotion_vis_client/interfaces/modules/principal_menu/principal_menu_ui_controller.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';

class DatasetTile extends GetView<PrincipalMenuUiController> {
  final String datasetId;
  const DatasetTile({Key key, this.datasetId}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: GestureDetector(
            onTap: () {
              controller.selectDataset(datasetId);
            },
            child: Container(
              height: 40,
              color: pColorAccent.withAlpha(50),
              alignment: Alignment.center,
              child: Text(datasetId),
            ),
          ),
        ),
        IconButton(
          icon: Icon(Icons.delete),
          onPressed: () {
            controller.removeDataset(datasetId);
          },
        ),
        IconButton(
          icon: Icon(Icons.settings),
          onPressed: () {
            controller.preprocessDataset(datasetId);
          },
        )
      ],
    );
  }
}
