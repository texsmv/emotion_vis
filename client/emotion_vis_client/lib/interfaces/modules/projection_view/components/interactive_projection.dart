import 'package:emotion_vis_client/controllers/series_controller.dart';
import 'package:emotion_vis_client/interfaces/constants/colors.dart';
import 'package:emotion_vis_client/interfaces/modules/home/home_ui_controller.dart';
import 'package:emotion_vis_client/interfaces/modules/projection_view/projection_view_ui_controller.dart';
import 'package:emotion_vis_client/interfaces/modules/visualizations_view/visualization_view_ui_controller.dart';
import 'package:emotion_vis_client/routes/route_names.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:glass_kit/glass_kit.dart';

class InteractiveProjection extends GetView<ProjectionViewUiController> {
  const InteractiveProjection({Key key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Container(
      height: double.infinity,
      width: double.infinity,
      child: Stack(
        children: [
          Positioned.fill(
            child: Padding(
              padding: const EdgeInsets.only(
                  left: 25, right: 10, top: 10, bottom: 10),
              child: LayoutBuilder(
                builder: (layoutContext, constraints) {
                  controller.windowWidth = constraints.maxWidth;
                  controller.windowHeigth = constraints.maxHeight;
                  controller.projectPointsToPlot();
                  controller.update();
                  return Listener(
                    onPointerUp: controller.onPointerUp,
                    onPointerDown: controller.onPointerDown,
                    onPointerMove: controller.onPointerMove,
                    child: Container(
                      color: Colors.white,
                      height: double.infinity,
                      width: double.infinity,
                      child: GetBuilder<SeriesController>(
                        builder: (_) => GetBuilder<ProjectionViewUiController>(
                          builder: (_) {
                            controller.updatePoints();
                            return Stack(
                              clipBehavior: Clip.none,
                              overflow: Overflow.visible,
                              children: [
                                Obx(
                                  () => Positioned(
                                    left: controller.selectionHorizontalStart,
                                    top: controller.selectionVerticalStart,
                                    child: Visibility(
                                      visible: controller.allowSelection,
                                      child: Container(
                                        color: Colors.blue.withAlpha(120),
                                        width: controller.selectionWidth,
                                        height: controller.selectionHeight,
                                        // width: 100,
                                        // height: 100,
                                      ),
                                    ),
                                  ),
                                ),
                                Obx(
                                  () => Positioned(
                                    left: controller.infoXposition.value +
                                        12 -
                                        80,
                                    top: controller.infoYposition.value -
                                        controller.infoHeight.value -
                                        8,
                                    child: controller.showInfo.value == true
                                        ? GlassContainer.clearGlass(
                                            borderRadius:
                                                BorderRadius.circular(15),
                                            padding: const EdgeInsets.all(10),
                                            borderColor: Colors.transparent,
                                            color:
                                                pColorPrimary.withOpacity(0.5),
                                            width: 160,
                                            height: controller.infoHeight.value,
                                            child: Column(
                                              children: [
                                                RichText(
                                                  text: TextSpan(
                                                    children: [
                                                      const TextSpan(
                                                        text: "Id: ",
                                                        style: TextStyle(
                                                          fontWeight:
                                                              FontWeight.w600,
                                                        ),
                                                      ),
                                                      TextSpan(
                                                        text: controller
                                                            .hoveredPoint
                                                            .personModel
                                                            .id,
                                                      )
                                                    ],
                                                    style: const TextStyle(
                                                      color: pTextColorWhite,
                                                      fontSize: 16,
                                                    ),
                                                  ),
                                                ),
                                                if (controller.datasetSettings
                                                    .hasCategoricalMetadata)
                                                  ListView.builder(
                                                    shrinkWrap: true,
                                                    itemCount: controller
                                                        .datasetSettings
                                                        .categoricalLabels
                                                        .length,
                                                    itemBuilder: (_, index) =>
                                                        RichText(
                                                      text: TextSpan(
                                                        style: const TextStyle(
                                                          color:
                                                              pTextColorWhite,
                                                          fontSize: 14,
                                                        ),
                                                        children: [
                                                          TextSpan(
                                                            text:
                                                                "${controller.datasetSettings.categoricalLabels[index]}: ",
                                                            style:
                                                                const TextStyle(
                                                              fontWeight:
                                                                  FontWeight
                                                                      .w500,
                                                            ),
                                                          ),
                                                          TextSpan(
                                                            text: controller
                                                                    .hoveredPoint
                                                                    .personModel
                                                                    .categoricalValues[
                                                                index],
                                                          )
                                                        ],
                                                      ),
                                                    ),
                                                  )
                                                else
                                                  const SizedBox(),
                                                if (controller.datasetSettings
                                                    .hasNumericalMetadata)
                                                  ListView.builder(
                                                    shrinkWrap: true,
                                                    itemCount: controller
                                                        .datasetSettings
                                                        .numericalLabels
                                                        .length,
                                                    itemBuilder: (_, index) =>
                                                        RichText(
                                                      text: TextSpan(
                                                        style: const TextStyle(
                                                          color:
                                                              pTextColorWhite,
                                                          fontSize: 14,
                                                        ),
                                                        children: [
                                                          TextSpan(
                                                              text:
                                                                  "${controller.datasetSettings.numericalLabels[index]}: "),
                                                          TextSpan(
                                                            text: controller
                                                                .hoveredPoint
                                                                .personModel
                                                                .numericalValues[
                                                                    index]
                                                                .toString(),
                                                          )
                                                        ],
                                                      ),
                                                    ),
                                                  )
                                                else
                                                  const SizedBox(),
                                              ],
                                            ),
                                          )
                                        : const SizedBox(),
                                  ),
                                ),
                              ]..addAll(
                                  List.generate(
                                    controller.points.length,
                                    (index) => ProjectionPoint(
                                      index: index,
                                    ),
                                  ),
                                ),
                            );
                          },
                        ),
                      ),
                    ),
                  );
                },
              ),
            ),
          ),
          const Positioned.fill(
            child: ClusterList(),
          )
        ],
      ),
    );
  }
}

class ProjectionPoint extends GetView<ProjectionViewUiController> {
  final int index;
  const ProjectionPoint({
    Key key,
    @required this.index,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return AnimatedPositioned(
      duration: const Duration(seconds: 1),
      left: controller.points[index].plotCoordinates.item1,
      top: controller.points[index].plotCoordinates.item2,
      child: MouseRegion(
        onHover: (event) {
          controller.hoveredPoint = controller.points[index];
          controller.infoXposition.value =
              controller.points[index].plotCoordinates.item1;
          controller.infoYposition.value =
              controller.points[index].plotCoordinates.item2;
          controller.showInfo.value = true;
        },
        onExit: (event) {
          controller.showInfo.value = false;
        },
        child: GestureDetector(
          onTap: () {
            Get.toNamed(routeSinglePerson,
                arguments: [controller.points[index].personModel]);
          },
          child: Container(
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: controller.points[index].personModel.clusterId == null
                  ? Colors.black
                  : controller.points[index].personModel.cluster.color
                      .withOpacity(0.8),
            ),
            width: 12,
            height: 12,
          ),
        ),
      ),
    );
  }
}

class ClusterList extends GetView<ProjectionViewUiController> {
  const ClusterList({Key key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: Alignment.centerLeft,
      child: Container(
        alignment: Alignment.topLeft,
        width: 150,
        padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 10),
        child: GetBuilder<SeriesController>(
          builder: (_) => ListView.separated(
            itemCount: controller.clustersIds.length,
            shrinkWrap: true,
            separatorBuilder: (context, index) => const SizedBox(height: 10),
            itemBuilder: (context, index) {
              return InkWell(
                onTap: () {
                  Get.find<VisualizationsViewUiController>()
                      .selectCluster(controller.clustersIds[index]);
                  Get.find<HomeUiController>().stackIndex = 0;
                },
                child: GlassContainer.frostedGlass(
                  borderColor: Colors.transparent,
                  margin: const EdgeInsets.symmetric(horizontal: 4),
                  width: 120,
                  height: 25,
                  child: Row(
                    children: [
                      Container(
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(2),
                          color: controller
                              .clusters[controller.clustersIds[index]].color,
                        ),
                        width: 13,
                        height: 13,
                      ),
                      const SizedBox(width: 10),
                      Text(controller
                          .clusters[controller.clustersIds[index]].id),
                      const SizedBox(width: 4),
                      Text(
                        "(${controller.clusters[controller.clustersIds[index]].persons.length})",
                        style: const TextStyle(
                          color: pTextColorSecondary,
                          fontSize: 12,
                        ),
                      ),
                    ],
                  ),
                ),
              );
            },
          ),
        ),
      ),
    );
  }
}
