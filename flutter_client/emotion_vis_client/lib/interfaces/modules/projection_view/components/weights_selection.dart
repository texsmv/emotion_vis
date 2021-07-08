import 'package:emotion_vis_client/interfaces/constants/colors.dart';
import 'package:emotion_vis_client/interfaces/general_widgets/containers/pexpanded_card.dart';
import 'package:emotion_vis_client/interfaces/modules/projection_view/projection_view_ui_controller.dart';
import 'package:flutter/material.dart';
import 'package:flutter_xlider/flutter_xlider.dart';
import 'package:get/get.dart';

class WeigthsSelection extends GetView<ProjectionViewUiController> {
  const WeigthsSelection({Key key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return PExpandedCard(
      header: const Text("Weigths"),
      content: SizedBox(
        height: 120,
        width: double.infinity,
        child: ListView.builder(
          scrollDirection: Axis.horizontal,
          itemCount: controller.datasetSettings.variablesLength,
          itemBuilder: (context, index) {
            return SizedBox(
              width: 100,
              child: Column(
                children: [
                  Text(
                    controller.datasetSettings.variablesNames[index],
                    style: const TextStyle(
                      fontWeight: FontWeight.w500,
                      color: pTextColorPrimary,
                      fontSize: 15,
                    ),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 5),
                  Expanded(
                    child: FlutterSlider(
                      values: [
                        controller.datasetSettings.alphas[controller
                                .datasetSettings.variablesNames[index]] *
                            100
                      ],
                      min: 0,
                      max: 100,
                      rtl: true,
                      touchSize: 20,
                      handlerHeight: 20,
                      handlerWidth: 20,
                      onDragCompleted: (handlerIndex, lowerValue, upperValue) {
                        double value = lowerValue / 100;
                        controller.datasetSettings.alphas[controller
                            .datasetSettings.variablesNames[index]] = value;
                        //     (upperValue - lowerValue) / handlerIndex;
                        controller.updateProjections();
                      },
                      handler: FlutterSliderHandler(
                        child: Material(
                          shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(10)),
                          color: Colors.orange,
                          elevation: 1,
                          child: Container(
                            decoration: const BoxDecoration(
                              shape: BoxShape.circle,
                              color: Colors.orange,
                            ),
                          ),
                        ),
                      ),
                      axis: Axis.vertical,
                      trackBar: FlutterSliderTrackBar(
                        inactiveTrackBar: BoxDecoration(
                          borderRadius: BorderRadius.circular(20),
                          color: Colors.black12,
                          border: Border.all(
                            width: 3,
                            color: Colors.grey,
                          ),
                        ),
                        activeTrackBar: BoxDecoration(
                          borderRadius: BorderRadius.circular(4),
                          color: Colors.orange,
                        ),
                      ),
                      tooltip: FlutterSliderTooltip(
                        disabled: true,
                      ),
                    ),
                  ),
                ],
              ),
            );
          },
        ),
      ),
    );
  }
}
