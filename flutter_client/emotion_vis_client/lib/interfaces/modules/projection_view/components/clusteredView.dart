import 'package:emotion_vis_client/controllers/series_controller.dart';
import 'package:emotion_vis_client/interfaces/constants/colors.dart';
import 'package:emotion_vis_client/interfaces/general_widgets/containers/pcard.dart';
import 'package:emotion_vis_client/interfaces/modules/projection_view/projection_view_ui_controller.dart';
import 'package:emotion_vis_client/interfaces/ui_utils.dart';
import 'package:emotion_vis_client/interfaces/visualizations/comparison/bars/grouped_bars.dart';
import 'package:emotion_vis_client/interfaces/visualizations/comparison/scatterplot/multi_dimensional_scatterplot.dart';
import 'package:emotion_vis_client/interfaces/visualizations/single_temporal/cluster_linear_chart/cluster_linear_chart.dart';
import 'package:emotion_vis_client/interfaces/visualizations/vis_settings.dart';
import 'package:emotion_vis_client/models/dataset_settings.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';

import 'clusters_selection.dart';

class ClusteredView extends GetView<ProjectionViewUiController> {
  const ClusteredView({Key key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final SeriesController _seriesController = Get.find();
    return GetBuilder<SeriesController>(
      builder: (_) => GetBuilder<ProjectionViewUiController>(
        builder: (_) => SizedBox(
          child: Column(
            children: [
              const SizedBox(height: 20),
              PCard(
                child: Container(
                  height: 60,
                  width: double.infinity,
                  child: ClustersSelection(),
                ),
              ),
              const SizedBox(height: 10),
              Expanded(
                flex: 3,
                child: PCard(
                  child: GetBuilder<SeriesController>(
                      builder: (_) => ClustersOverview()),
                ),
              ),
              const SizedBox(height: 10),
              Expanded(
                flex: 5,
                child: PCard(
                  child: Column(
                    children: [
                      const Text(
                        "Emotion dimensions ranking",
                        style: TextStyle(
                          fontSize: 16,
                          color: pColorAccent,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      Expanded(
                        child: GetBuilder<SeriesController>(builder: (_) {
                          if (controller.clusterIdA == null ||
                              controller.clusterIdB == null) {
                            return const Center(
                              child: Text(
                                "Select two clusters to compare",
                                style: TextStyle(
                                  fontSize: 14,
                                  color: pColorGray,
                                  fontWeight: FontWeight.w400,
                                ),
                              ),
                            );
                          }
                          return (_seriesController.variablesNamesOrdered !=
                                      null &&
                                  controller.areStatsLoaded)
                              ? Scrollbar(
                                  child: ListView.separated(
                                    separatorBuilder: (context, index) =>
                                        const Divider(
                                      color: pColorLight,
                                      thickness: 1,
                                    ),
                                    itemCount:
                                        controller.variablesOrdered.length,
                                    itemBuilder: (context, index) {
                                      return Padding(
                                        padding: const EdgeInsets.symmetric(
                                            horizontal: 10, vertical: 10),
                                        child: Column(
                                          crossAxisAlignment:
                                              CrossAxisAlignment.start,
                                          children: [
                                            AspectRatio(
                                              aspectRatio: 16 / 6.5,
                                              child: Row(
                                                children: [
                                                  RotatedBox(
                                                    quarterTurns: 3,
                                                    child: Container(
                                                      height: 16,
                                                      child: Text(
                                                        controller
                                                                .variablesOrdered[
                                                            index],
                                                        style: TextStyle(
                                                          fontSize: 16,
                                                          fontWeight:
                                                              FontWeight.w500,
                                                        ),
                                                      ),
                                                    ),
                                                  ),
                                                  const SizedBox(width: 15),
                                                  Expanded(
                                                    child: ClusterLinearChart(
                                                      visSettings: VisSettings(
                                                        upperLimit: uiUtilMapMax(
                                                            controller
                                                                .datasetSettings
                                                                .maxValues),
                                                        lowerLimit: uiUtilMapMin(
                                                            controller
                                                                .datasetSettings
                                                                .minValues),
                                                      ),
                                                      blueClusterColor: controller
                                                          .clusters[controller
                                                              .clusterIdA]
                                                          .color,
                                                      redClusterColor: controller
                                                          .clusters[controller
                                                              .clusterIdB]
                                                          .color,
                                                      clusterAmeans: List<
                                                          double>.from(controller
                                                              .clusterATemporalStats[
                                                          "mean"][controller
                                                              .variablesOrdered[
                                                          index]]),
                                                      clusterAstd: List<
                                                          double>.from(controller
                                                              .clusterATemporalStats[
                                                          "std"][controller
                                                              .variablesOrdered[
                                                          index]]),
                                                      clusterBmeans: List<
                                                          double>.from(controller
                                                              .clusterBTemporalStats[
                                                          "mean"][controller
                                                              .variablesOrdered[
                                                          index]]),
                                                      clusterBstd: List<
                                                          double>.from(controller
                                                              .clusterBTemporalStats[
                                                          "std"][controller
                                                              .variablesOrdered[
                                                          index]]),
                                                      variableName: controller
                                                              .variablesOrdered[
                                                          index],
                                                    ),
                                                  ),
                                                ],
                                              ),
                                            ),
                                          ],
                                        ),
                                      );
                                    },
                                  ),
                                )
                              : const Center(
                                  child: CircularProgressIndicator(),
                                );
                        }),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class ClustersOverview extends GetView<ProjectionViewUiController> {
  const ClustersOverview({Key key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    Widget child;
    if (controller.clusterIdA == null || controller.clusterIdB == null) {
      child = SizedBox.expand(
        child: const Center(
            child: Text(
          "Select two clusters to compare",
          style: TextStyle(
            fontSize: 14,
            color: pColorGray,
            fontWeight: FontWeight.w400,
          ),
        )),
      );
    } else if (!controller.areStatsLoaded) {
      child = const Center(
        child: CircularProgressIndicator(),
      );
    } else if (controller.datasetSettings.modelType == ModelType.DIMENSIONAL) {
      child = Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          // Expanded(child: SizedBox()),
          Expanded(
            child: Column(
              children: [
                Expanded(
                  child: Padding(
                    padding: const EdgeInsets.symmetric(vertical: 20),
                    child: MultiDimensionalScatterplot(
                      visSettings: VisSettings(
                        upperLimit:
                            uiUtilMapMax(controller.datasetSettings.maxValues),
                        lowerLimit:
                            uiUtilMapMin(controller.datasetSettings.minValues),
                      ),
                      colorA: controller.clusters[controller.clusterIdA].color,
                      colorB: controller.clusters[controller.clusterIdB].color,
                      clusterA: controller.blueCluster,
                      clusterB: controller.redCluster,
                      nSideBins: 20,
                    ),
                  ),
                ),
                SizedBox(
                  height: 18,
                )
              ],
            ),
          ),
          // Expanded(child: SizedBox()),
        ],
      );
    } else {
      child = Container(
          child: GroupedBars(
        colorA: controller.clusters[controller.clusterIdA].color,
        colorB: controller.clusters[controller.clusterIdB].color,
        averagesA: List.generate(
            controller.datasetSettings.variablesNames.length,
            (index) => controller.clusterAInstantStats["mean"]
                [controller.datasetSettings.variablesNames[index]]),
        averagesB: List.generate(
            controller.datasetSettings.variablesNames.length,
            (index) => controller.clusterBInstantStats["mean"]
                [controller.datasetSettings.variablesNames[index]]),
        // clusterA: controller.blueCluster,
        // clusterB: controller.redCluster,
      ));
    }

    return Column(
      children: [
        const Text(
          "Summary",
          style: TextStyle(
            fontSize: 16,
            color: pColorAccent,
            fontWeight: FontWeight.w600,
          ),
        ),
        Expanded(child: child),
      ],
    );
  }
}
