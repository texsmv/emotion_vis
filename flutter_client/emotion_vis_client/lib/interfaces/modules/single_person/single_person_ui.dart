import 'package:emotion_vis_client/app_constants.dart';
import 'package:emotion_vis_client/controllers/series_controller.dart';
import 'package:emotion_vis_client/interfaces/constants/colors.dart';
import 'package:emotion_vis_client/interfaces/general_widgets/containers/pcard.dart';
import 'package:emotion_vis_client/interfaces/general_widgets/fields/pinfo_text.dart';
import 'package:emotion_vis_client/interfaces/general_widgets/options_bar/Options_bar.dart';
import 'package:emotion_vis_client/interfaces/general_widgets/side_bar/side_bar.dart';
import 'package:emotion_vis_client/interfaces/mod_packages/pflutter_xlider.dart';
import 'package:emotion_vis_client/interfaces/modules/single_person/components/time_progress.dart';
import 'package:emotion_vis_client/interfaces/modules/single_person/single_person_ui_controller.dart';
import 'package:emotion_vis_client/interfaces/ui_utils.dart';
import 'package:emotion_vis_client/interfaces/visualizations/non_temporal/non_temporal_chart.dart';
import 'package:emotion_vis_client/interfaces/visualizations/temporal/temporal_chart.dart';
import 'package:emotion_vis_client/interfaces/visualizations/vis_settings.dart';
import 'package:emotion_vis_client/models/dataset_settings.dart';
import 'package:emotion_vis_client/models/person_model.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';

class SinglePersonUi extends GetView<SinglePersonUiController> {
  final PersonModel person;
  SinglePersonUi({Key key, @required this.person}) : super(key: key);

  final SeriesController _seriesController = Get.find();
  DatasetSettings get datasetSettings => _seriesController.datasetSettings;

  @override
  Widget build(BuildContext context) {
    controller.initState(person);

    List<Widget> _options() {
      final List<Widget> options = [];
      options.add(
        Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            Padding(
              padding: EdgeInsets.symmetric(vertical: 10, horizontal: 5),
              child: Text(
                "Temporal chart:".toUpperCase(),
                style: TextStyle(
                  fontWeight: FontWeight.w600,
                  fontSize: 15,
                ),
                textAlign: TextAlign.center,
              ),
            ),
            ListView.separated(
              physics: const NeverScrollableScrollPhysics(),
              separatorBuilder: (context, index) => const SizedBox(height: 5),
              itemCount: controller.availableTemporalVisualizations.length,
              shrinkWrap: true,
              itemBuilder: (context, index) {
                return InkWell(
                  onTap: () {
                    controller.temporalVisualization =
                        controller.availableTemporalVisualizations[index];
                    controller.update();
                  },
                  child: Container(
                    height: 40,
                    decoration: BoxDecoration(
                      color:
                          controller.availableTemporalVisualizations[index] ==
                                  controller.temporalVisualization
                              ? pColorAccent.withOpacity(0.7)
                              : const Color.fromARGB(0, 240, 240, 240),
                    ),
                    alignment: Alignment.centerLeft,
                    padding: const EdgeInsets.symmetric(horizontal: 10),
                    child: Text(
                        uiUtilTemVis2Str(
                            controller.availableTemporalVisualizations[index]),
                        textAlign: TextAlign.start,
                        overflow: TextOverflow.ellipsis,
                        style: TextStyle(
                          color: controller
                                      .availableTemporalVisualizations[index] ==
                                  controller.temporalVisualization
                              ? Colors.white
                              : Colors.black,
                        )),
                  ),
                );
              },
            ),
          ],
        ),
      );

      options.add(
        Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            Padding(
              padding: EdgeInsets.symmetric(vertical: 10, horizontal: 5),
              child: Text(
                "Non temporal chart:".toUpperCase(),
                style: TextStyle(
                  fontWeight: FontWeight.w600,
                  fontSize: 15,
                ),
                textAlign: TextAlign.center,
              ),
            ),
            ListView.separated(
              physics: const NeverScrollableScrollPhysics(),
              separatorBuilder: (context, index) => const SizedBox(height: 5),
              itemCount: controller.availableNonTemporalVisualizations.length,
              shrinkWrap: true,
              itemBuilder: (context, index) {
                return InkWell(
                  onTap: () {
                    controller.nonTemporalVisualization =
                        controller.availableNonTemporalVisualizations[index];
                    controller.update();
                  },
                  child: Container(
                    height: 40,
                    decoration: BoxDecoration(
                      color: controller
                                  .availableNonTemporalVisualizations[index] ==
                              controller.nonTemporalVisualization
                          ? pColorAccent.withOpacity(0.7)
                          : const Color.fromARGB(0, 240, 240, 240),
                    ),
                    alignment: Alignment.centerLeft,
                    padding: const EdgeInsets.symmetric(horizontal: 10),
                    child: Text(
                        uiUtilNonTemVis2Str(controller
                            .availableNonTemporalVisualizations[index]),
                        textAlign: TextAlign.start,
                        overflow: TextOverflow.ellipsis,
                        style: TextStyle(
                          color: controller.availableNonTemporalVisualizations[
                                      index] ==
                                  controller.nonTemporalVisualization
                              ? Colors.white
                              : Colors.black,
                        )),
                  ),
                );
              },
            ),
          ],
        ),
      );

      // * Adding dateformat options
      if (controller.datasetSettings.isDated) {
        options.add(
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Padding(
                padding: EdgeInsets.symmetric(vertical: 10, horizontal: 5),
                child: Text(
                  "Date format:".toUpperCase(),
                  style: TextStyle(
                    fontWeight: FontWeight.w600,
                    fontSize: 15,
                  ),
                  textAlign: TextAlign.center,
                ),
              ),
              ListView.separated(
                physics: const NeverScrollableScrollPhysics(),
                separatorBuilder: (context, index) => const SizedBox(height: 5),
                itemCount: DateStrFormat.values.length,
                shrinkWrap: true,
                itemBuilder: (context, index) {
                  return InkWell(
                    onTap: () {
                      Get.find<SeriesController>().updateSettings(
                          dateFormat: DateStrFormat.values[index]);
                      Get.back();
                    },
                    child: Container(
                      height: 40,
                      decoration: BoxDecoration(
                        color: controller.datasetSettings.dateFormat ==
                                DateStrFormat.values[index]
                            ? pColorAccent.withOpacity(0.7)
                            : const Color.fromARGB(0, 240, 240, 240),
                      ),
                      alignment: Alignment.centerLeft,
                      padding: const EdgeInsets.symmetric(horizontal: 10),
                      child: Text(
                        uiUtilDateFormatToStr(
                          DateStrFormat.values[index],
                        ),
                        textAlign: TextAlign.start,
                        overflow: TextOverflow.ellipsis,
                        style: TextStyle(
                          color: controller.datasetSettings.dateFormat ==
                                  DateStrFormat.values[index]
                              ? Colors.white
                              : Colors.black,
                        ),
                      ),
                    ),
                  );
                },
              ),
            ],
          ),
        );
      }
      return options;
    }

    return SafeArea(
      child: Scaffold(
        body: Container(
          width: double.infinity,
          height: double.infinity,
          decoration: const BoxDecoration(
            image: DecorationImage(
              image: AssetImage(ASSET_WALLPAPER),
              fit: BoxFit.cover,
            ),
          ),
          child: GetBuilder<SinglePersonUiController>(
            builder: (_) => controller.personModel != null
                ? Row(
                    children: [
                      GetBuilder<SeriesController>(
                        builder: (_) => SideBar(
                          selectedTab: 0,
                          tabs: [
                            ViewTab(
                              icon: Icons.person,
                              onTap: () {},
                              text: "Visualizations",
                              options: _options(),
                            )
                          ],
                          actions: [
                            ActionButton(
                              icon: Icons.arrow_back,
                              onTap: () {
                                Get.back();
                              },
                            )
                          ],
                        ),
                      ),
                      content(),
                    ],
                  )
                : Container(
                    width: double.infinity,
                    height: double.infinity,
                    color: pColorScaffold,
                    // child: Center(
                    //   child: CircularProgressIndicator(),
                    // ),
                  ),
          ),
        ),
      ),
    );
  }

  Widget content() {
    if (!controller.personModel.isDataLoaded) {
      controller.personModel.loadEmotions().then((value) {
        controller.update();
      });
      return const Center(
        child: CircularProgressIndicator(),
      );
    }
    return Expanded(
      child: Container(
        color: pColorScaffold,
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 40),
              child: PCard(
                child: Container(
                  height: 60,
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
                        child: ListView.separated(
                          separatorBuilder: (context, index) =>
                              SizedBox(width: 20),
                          scrollDirection: Axis.horizontal,
                          itemCount: datasetSettings.variablesNames.length,
                          itemBuilder: (context, index) {
                            String variable =
                                datasetSettings.variablesNames[index];
                            Color color =
                                datasetSettings.variablesColors[variable];
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
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
            Expanded(
              child: Padding(
                padding: EdgeInsets.symmetric(vertical: 10, horizontal: 40),
                child: PCard(
                  child: Container(
                    // color: Colors.red,
                    // padding: EdgeInsets.symmetric(vertical: 10, horizontal: 40),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Expanded(
                          child: GetBuilder<SeriesController>(
                            builder: (_) =>
                                GetBuilder<SinglePersonUiController>(
                              builder: (_) => SizedBox(
                                width: double.infinity,
                                child: controller.personModel != null
                                    ? TemporalChart(
                                        modelType: controller
                                            .datasetSettings.modelType,
                                        temporalVisualization:
                                            controller.temporalVisualization,
                                        personModel: controller.personModel,
                                        visSettings: VisSettings(
                                          colors: controller
                                              .datasetSettings.variablesColors,
                                          timePoint: controller.timePoint,
                                          lowerLimits: controller
                                              .datasetSettings.minValues,
                                          upperLimits: controller
                                              .datasetSettings.maxValues,
                                          lowerLimit: uiUtilMapMin(controller
                                              .datasetSettings.minValues),
                                          upperLimit: uiUtilMapMax(controller
                                              .datasetSettings.maxValues),
                                        ),
                                      )
                                    : const Center(
                                        child: CircularProgressIndicator(),
                                      ),
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 10),
            Padding(
              padding: EdgeInsets.symmetric(horizontal: 40),
              child: PCard(child: TimeProgress()),
            ),
            const SizedBox(height: 10),
            Expanded(
              child: Container(
                child: Column(
                  children: [
                    Expanded(
                      child: Padding(
                        padding:
                            EdgeInsets.symmetric(vertical: 10, horizontal: 40),
                        child: Row(
                          children: [
                            Expanded(
                              child: PCard(
                                child: Container(
                                  child: Column(
                                    crossAxisAlignment:
                                        CrossAxisAlignment.start,
                                    children: [
                                      Text(
                                        "User information:",
                                        style: TextStyle(
                                          color: pTextColorPrimary,
                                          fontWeight: FontWeight.w600,
                                          fontSize: 16,
                                        ),
                                      ),
                                      SizedBox(height: 10),
                                      Padding(
                                        padding: EdgeInsets.symmetric(
                                            horizontal: 10),
                                        child: Container(
                                          width: 200,
                                          child: PInfoText(
                                            label: "id",
                                            text: controller.personId,
                                          ),
                                        ),
                                      ),
                                      SizedBox(height: 20),
                                      Text(
                                        "Categorical metadata",
                                        style: TextStyle(
                                          color: pTextColorPrimary,
                                          fontWeight: FontWeight.w600,
                                          fontSize: 16,
                                        ),
                                      ),
                                      SizedBox(height: 10),
                                      Wrap(
                                        children: List.generate(
                                          controller.personModel
                                              .categoricalLabels.length,
                                          (index) => Container(
                                            width: 200,
                                            padding: EdgeInsets.symmetric(
                                                horizontal: 10),
                                            child: PInfoText(
                                              label: controller.personModel
                                                  .categoricalLabels[index],
                                              text: controller.personModel
                                                  .categoricalValues[index],
                                            ),
                                          ),
                                        ),
                                      ),
                                      SizedBox(height: 20),
                                      Text(
                                        "Numerical metadata",
                                        style: TextStyle(
                                          color: pTextColorPrimary,
                                          fontWeight: FontWeight.w600,
                                          fontSize: 16,
                                        ),
                                      ),
                                      SizedBox(height: 10),
                                      Wrap(
                                        children: List.generate(
                                          controller.personModel.numericalLabels
                                              .length,
                                          (index) => Container(
                                            width: 200,
                                            padding: EdgeInsets.symmetric(
                                                horizontal: 10),
                                            child: PInfoText(
                                              label: controller.personModel
                                                  .numericalLabels[index],
                                              text: controller.personModel
                                                  .numericalValues[index]
                                                  .toString(),
                                            ),
                                          ),
                                        ),
                                      )
                                    ],
                                  ),
                                ),
                              ),
                            ),
                            SizedBox(width: 20),
                            Expanded(
                              child: controller.personModel != null
                                  ? PCard(
                                      child: Container(
                                        height: double.infinity,
                                        width: double.infinity,
                                        child: Column(
                                          mainAxisAlignment:
                                              MainAxisAlignment.start,
                                          children: [
                                            Text(datasetSettings.allLabels[
                                                controller.timePoint]),
                                            Expanded(
                                              child: Container(
                                                height: double.infinity,
                                                width: double.infinity,
                                                child: NonTemporalChart(
                                                  personModel:
                                                      controller.personModel,
                                                  modelType: controller
                                                      .datasetSettings
                                                      .modelType,
                                                  nonTemporalVisualization:
                                                      controller
                                                          .nonTemporalVisualization,
                                                  timePoint:
                                                      controller.timePoint,
                                                  visSettings: VisSettings(
                                                    timePoint:
                                                        controller.timePoint,
                                                    colors: controller
                                                        .datasetSettings
                                                        .variablesColors,
                                                    lowerLimits: controller
                                                        .datasetSettings
                                                        .minValues,
                                                    upperLimits: controller
                                                        .datasetSettings
                                                        .maxValues,
                                                    lowerLimit: uiUtilMapMin(
                                                        controller
                                                            .datasetSettings
                                                            .minValues),
                                                    upperLimit: uiUtilMapMax(
                                                        controller
                                                            .datasetSettings
                                                            .maxValues),
                                                    // variablesNames:
                                                    //     controller
                                                    //         .datasetSettings
                                                    //         .variablesNames,
                                                  ),
                                                ),
                                              ),
                                            ),
                                          ],
                                        ),
                                      ),
                                    )
                                  : const Center(
                                      child: CircularProgressIndicator(),
                                    ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            )
          ],
        ),
      ),
    );
  }
}
