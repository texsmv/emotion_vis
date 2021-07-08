import 'package:emotion_vis_client/interfaces/constants/colors.dart';
import 'package:emotion_vis_client/interfaces/general_widgets/buttons/pfilled_button.dart';
import 'package:emotion_vis_client/interfaces/ui_utils.dart';
import 'package:flutter/material.dart';
import 'package:flutter_icons/flutter_icons.dart';
import 'package:get/get.dart';
import 'package:prefab_animations/event_animation/animations/bounce_animation.dart';
import 'package:prefab_animations/event_animation/event_animation.dart';

import '../principal_menu_ui_controller.dart';

class DatasetSelectionSide extends GetView<PrincipalMenuUiController> {
  const DatasetSelectionSide({Key key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Container(
      color: Colors.white,
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Text(
            "Dataset selection",
            style: TextStyle(
              fontSize: 26,
              fontWeight: FontWeight.w600,
              color: pColorAccent,
            ),
          ),
          const SizedBox(height: 20),
          GetBuilder<PrincipalMenuUiController>(
            builder: (_) => SizedBox(
              // width: 400,
              height: 400,
              child: controller.datasetsSettings != null
                  ? controller.datasetsSettings.keys.isNotEmpty
                      ? DataTable(
                          showCheckboxColumn: false,
                          columns: const [
                            DataColumn(
                              label: Text("id"),
                            ),
                            DataColumn(
                              label: Text("type"),
                            ),
                            DataColumn(
                              label: Text("Nº variables"),
                            ),
                            DataColumn(
                              label: Text("Nº instances"),
                            ),
                            DataColumn(
                              label: Text("Nº time points"),
                            ),
                            DataColumn(
                              label: Text("Options"),
                            ),
                            // DataColumn(label: Text("type")),
                          ],
                          rows: List.generate(
                            controller.loadedDatasetsIds.length,
                            (index) => _dataRowById(
                                controller.loadedDatasetsIds[index]),
                          ),
                        )
                      : const Center(
                          child: Text("No datasets loaded"),
                        )
                  : const Center(
                      child: CircularProgressIndicator(),
                    ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.end,
              children: [
                GetBuilder<PrincipalMenuUiController>(
                  builder: (_) => controller.nLoadedDatasets == 0
                      ? EventAnimation(
                          awaitAnimationDuration:
                              const Duration(milliseconds: 600),
                          awaitAnimationBuilder: (controller, child) {
                            return BounceAnimation(
                              controller: controller,
                              child: child,
                              minScale: 0.90,
                            );
                          },
                          child: SizedBox(
                            width: 120,
                            child: PFilledButton(
                              buttonColor: pColorDark,
                              text: "Load dataset",
                              onPressed: controller.openLocalDataset,
                            ),
                          ),
                        )
                      : SizedBox(
                          width: 120,
                          child: PFilledButton(
                            buttonColor: pColorDark,
                            text: "Load dataset",
                            onPressed: controller.openLocalDataset,
                          ),
                        ),
                ),
                // TODO enable later
                const SizedBox(width: 20),
                SizedBox(
                  width: 110,
                  child: PFilledButton(
                    text: "Add dataset",
                    onPressed: controller.addDataset,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  DataRow _dataRowById(String id) {
    return DataRow(
      onSelectChanged: (value) {
        controller.selectDataset(id);
      },
      cells: [
        DataCell(
          Text(id),
        ),
        DataCell(
          Text(uiUtilModelTypeToStr(controller.datasetsSettings[id].modelType)),
        ),
        DataCell(
          Text(controller.datasetsSettings[id].variablesLength.toString()),
        ),
        DataCell(
          Text(controller.datasetsSettings[id].instanceLength.toString()),
        ),
        DataCell(
          Text(controller.datasetsSettings[id].timeLength.toString()),
        ),
        DataCell(
          Row(
            children: [
              IconButton(
                icon: Icon(MaterialIcons.date_range),
                onPressed: () {
                  controller.preprocessDataset(id);
                },
              ),
              IconButton(
                icon: Icon(Icons.delete),
                color: Colors.red,
                onPressed: () {
                  controller.removeDataset(id);
                },
              ),
            ],
          ),
        ),
      ],
    );
  }
}
