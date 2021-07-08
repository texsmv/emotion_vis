import 'package:emotion_vis_client/controllers/series_controller.dart';
import 'package:emotion_vis_client/interfaces/constants/colors.dart';
import 'package:emotion_vis_client/interfaces/general_widgets/containers/pcard.dart';
import 'package:emotion_vis_client/interfaces/general_widgets/containers/pexpanded_card.dart';
import 'package:emotion_vis_client/interfaces/general_widgets/others/options_list.dart';
import 'package:emotion_vis_client/interfaces/mod_packages/pflutter_xlider.dart';
import 'package:emotion_vis_client/interfaces/modules/home/home_ui_controller.dart';
import 'package:emotion_vis_client/interfaces/ui_utils.dart';
import 'package:emotion_vis_client/interfaces/visualizations/overview/stacked_bar_chart.dart/stacked_bar_chart.dart';
import 'package:emotion_vis_client/interfaces/visualizations/vis_settings.dart';
import 'package:emotion_vis_client/models/MTSerie.dart';
import 'package:emotion_vis_client/models/dataset_settings.dart';
import 'package:emotion_vis_client/models/visualization_levels.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
// import 'package:flutter_xlider/flutter_xlider.dart';
import 'package:syncfusion_flutter_sliders/sliders.dart';

class SummaryVisualizationView extends GetView<HomeUiController> {
  const SummaryVisualizationView({Key key}) : super(key: key);
  DatasetSettings get datasetSettings => controller.datasetSettings;

  @override
  Widget build(BuildContext context) {
    return GetBuilder<SeriesController>(
      builder: (_) => PExpandedCard(
        header: Row(
          children: [
            Text("Overview"),
            SizedBox(width: 10),
            Expanded(
              child: Visibility(
                visible: !Get.find<SeriesController>().visualizeAllTime,
                child: PFlutterSlider(
                  values: [
                    datasetSettings.begin.toDouble(),
                    datasetSettings.end.toDouble()
                  ],
                  touchSize: 0,
                  visibleTouchArea: false,
                  rangeSlider: true,
                  max: datasetSettings.timeLength.toDouble(),
                  min: 0,
                  minimumDistance: datasetSettings.windowLength.toDouble(),
                  maximumDistance: datasetSettings.windowLength.toDouble() + 1,
                  onDragging: (handlerIndex, lowerValue, upperValue) {
                    datasetSettings.windowPosition = lowerValue.toInt();
                    datasetSettings.end = datasetSettings.windowPosition +
                        datasetSettings.windowLength;
                  },
                  onDragCompleted: (handlerIndex, lowerValue, upperValue) {
                    controller.updateRange();
                  },
                  handlerHeight: 3,
                  handlerWidth: 3,
                  handler: FlutterSliderHandler(child: SizedBox()),
                  rightHandler: FlutterSliderHandler(child: SizedBox()),
                  trackBar: FlutterSliderTrackBar(
                    activeTrackBarHeight: 20,
                    inactiveTrackBarHeight: 15,
                    // inactiveTrackBar: BoxDecoration(
                    //   borderRadius: BorderRadius.circular(20),
                    //   color: Colors.black12,
                    //   border: Border.all(width: 10, color: Colors.blue),
                    // ),
                    activeTrackBar: BoxDecoration(
                        borderRadius: BorderRadius.circular(4),
                        color: pColorAccent.withOpacity(0.5)),
                  ),
                ),
              ),
            )
          ],
        ),
        content: Column(
          children: [
            SizedBox(height: 20),
            OptionsList(
              optionsTitles: List.generate(
                OverviewType.values.length,
                (index) => uiUtilOverviewTypeToStr(OverviewType.values[index]),
              ),
              onSelected: (index) {
                controller.datasetSettings.overviewType =
                    OverviewType.values[index];
                controller.update();
              },
            ),
            Container(
              height: 150,
              width: double.infinity,
              // decoration: BoxDecoration(
              //   color: pColorBackground,
              //   border: Border.all(color: pColorPrimary, width: 2),
              //   borderRadius: BorderRadius.circular(16),
              // ),
              child: GetBuilder<HomeUiController>(
                builder: (_) => StackedBarChart(
                  mtSerie: MTSerie.fromMap(controller.overview),
                  visSettings: VisSettings(
                    colors: controller.datasetSettings.variablesColors,
                    lowerLimit:
                        uiUtilMapMin(controller.datasetSettings.minValues),
                    upperLimit:
                        uiUtilMapMax(controller.datasetSettings.maxValues) *
                            controller.datasetSettings.variablesLength,
                    // variablesNames: controller.datasetSettings.variablesNames,
                    // timeLabels: controller.datasetSettings.timeLabels,
                  ),
                  begin: controller.datasetSettings.begin,
                  end: controller.datasetSettings.end,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
