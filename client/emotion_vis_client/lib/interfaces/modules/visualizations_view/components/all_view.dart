import 'package:auto_size_text/auto_size_text.dart';
import 'package:emotion_vis_client/controllers/series_controller.dart';
import 'package:emotion_vis_client/interfaces/constants/colors.dart';
import 'package:emotion_vis_client/interfaces/general_widgets/containers/pexpanded_card.dart';
import 'package:emotion_vis_client/interfaces/modules/home/components/summary_visualization_view.dart';
import 'package:emotion_vis_client/interfaces/modules/home/home_ui_controller.dart';
import 'package:emotion_vis_client/interfaces/modules/visualizations_view/visualization_view_ui_controller.dart';
import 'package:emotion_vis_client/interfaces/ui_utils.dart';
import 'package:emotion_vis_client/interfaces/visualizations/temporal/temporal_chart.dart';
import 'package:emotion_vis_client/interfaces/visualizations/vis_settings.dart';
import 'package:emotion_vis_client/models/dataset_settings.dart';
import 'package:emotion_vis_client/models/person_model.dart';
import 'package:emotion_vis_client/models/visualization_levels.dart';
import 'package:emotion_vis_client/routes/route_names.dart';
import 'package:flutter/material.dart';
import 'package:flutter_xlider/flutter_xlider.dart';
import 'package:get/get.dart';
import 'package:get/get_state_manager/src/simple/get_state.dart';

const double _tileHeight = 30;
const double _contentHeight = 300;

class AllView extends GetView<VisualizationsViewUiController> {
  AllView({Key key}) : super(key: key);
  DatasetSettings get datasetSettings => controller.datasetSettings;

  final HomeUiController _homeUiController = Get.find();

  @override
  Widget build(BuildContext context) {
    return Container(
      child: GetBuilder<SeriesController>(
        builder: (_) => GetBuilder<HomeUiController>(
          builder: (_) => GetBuilder<VisualizationsViewUiController>(
            builder: (_) => Column(
              children: [
                Expanded(
                  child: Column(
                    children: [
                      Container(
                        height: _tileHeight + 20,
                        color: const Color.fromARGB(255, 235, 235, 235),
                        padding: const EdgeInsets.symmetric(
                            horizontal: 20, vertical: 10),
                        child: Row(
                          children: [
                            Expanded(
                              child: ListView.builder(
                                itemCount:
                                    datasetSettings.identifiersLabels.length,
                                scrollDirection: Axis.horizontal,
                                itemBuilder: (context, index) => SizedBox(
                                  width: 200,
                                  child: Text(
                                    datasetSettings.identifiersLabels[index]
                                        .toUpperCase(),
                                    style: const TextStyle(
                                      color: pTextColorSecondary,
                                      fontSize: 16,
                                      fontWeight: FontWeight.w500,
                                    ),
                                  ),
                                ),
                              ),
                            ),
                            Container(
                              width: 50,
                              alignment: Alignment.center,
                              child: Text(
                                "Expand",
                                style: const TextStyle(
                                  fontSize: 15,
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                            ),
                            const SizedBox(width: 8),
                            VerticalDivider(
                              width: 3,
                            ),
                            const SizedBox(width: 5),
                            Container(
                              width: 50,
                              alignment: Alignment.center,
                              child: Text(
                                "Details",
                                style: const TextStyle(
                                  fontSize: 15,
                                  color: pColorAccent,
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                            ),
                            const SizedBox(width: 16),
                          ],
                        ),
                      ),
                      Expanded(
                        child: Scrollbar(
                          controller: controller.listScrollController,
                          child: ListView.separated(
                            controller: controller.listScrollController,
                            itemCount: controller.persons.length,
                            separatorBuilder: (context, index) => Divider(
                              height: 2,
                            ),
                            itemBuilder: (_, index) {
                              return Container(
                                child: PersonTile(
                                  person: controller.persons[index],
                                  datasetSettings: controller.datasetSettings,
                                  visSettings: VisSettings(
                                    colors: controller.colors,
                                    lowerLimits:
                                        controller.datasetSettings.minValues,
                                    upperLimits:
                                        controller.datasetSettings.maxValues,
                                    lowerLimit: uiUtilMapMin(
                                        controller.datasetSettings.minValues),
                                    upperLimit: uiUtilMapMax(
                                        controller.datasetSettings.maxValues),
                                    // variablesNames:
                                    //     controller.datasetSettings.variablesNames,
                                    // timeLabels: controller.datasetSettings.dates,
                                  ),
                                  temporalVisualization:
                                      controller.temporalVisualization,
                                ),
                              );
                            },
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
                SizedBox(height: 20),
                const SummaryVisualizationView(),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class PersonTile extends StatefulWidget {
  final PersonModel person;
  final DatasetSettings datasetSettings;
  final TemporalVisualization temporalVisualization;
  final VisSettings visSettings;
  const PersonTile({
    Key key,
    @required this.person,
    @required this.datasetSettings,
    @required this.visSettings,
    @required this.temporalVisualization,
  }) : super(key: key);

  @override
  _PersonTileState createState() => _PersonTileState();
}

class _PersonTileState extends State<PersonTile> {
  PersonModel get person => widget.person;
  DatasetSettings get datasetSettings => widget.datasetSettings;

  @override
  Widget build(BuildContext context) {
    return PExpandedCard(
      borderRadius: 0,
      header: Container(
        height: _tileHeight,
        // child: Text(person.id),
        child: ListView.builder(
          itemCount: datasetSettings.identifiersLabels.length,
          scrollDirection: Axis.horizontal,
          itemBuilder: (context, index) => SizedBox(
            width: 200,
            child: AutoSizeText(
              person.metadata[datasetSettings.identifiersLabels[index]],
              maxLines: 2,
              minFontSize: 8,
              style: const TextStyle(
                color: pTextColorSecondary,
                fontSize: 16,
                fontWeight: FontWeight.w500,
              ),
            ),
          ),
        ),
      ),
      onRedirect: () {
        Get.toNamed(routeSinglePerson, arguments: [person]);
      },
      content: Container(
        height: _contentHeight,
        width: double.infinity,
        child: _visualization(),
      ),
    );
  }

  Widget _visualization() {
    return TemporalChart(
      personModel: person,
      modelType: datasetSettings.modelType,
      temporalVisualization: widget.temporalVisualization,
      visSettings: widget.visSettings,
    );
  }
}
