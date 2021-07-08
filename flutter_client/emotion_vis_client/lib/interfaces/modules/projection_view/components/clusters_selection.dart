import 'package:emotion_vis_client/controllers/series_controller.dart';
import 'package:emotion_vis_client/interfaces/constants/colors.dart';
import 'package:emotion_vis_client/interfaces/modules/projection_view/projection_view_ui_controller.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:menu_button/menu_button.dart';

class ClustersSelection extends GetView<ProjectionViewUiController> {
  const ClustersSelection({Key key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        const Text(
          "Cluster selection",
          style: TextStyle(
            fontSize: 16,
            color: pColorAccent,
            fontWeight: FontWeight.w600,
          ),
        ),
        Expanded(
          child: controller.clusters.length >= 2
              ? SizedBox(
                  height: 40,
                  child: GetBuilder<SeriesController>(builder: (_) {
                    return Row(
                      mainAxisAlignment: MainAxisAlignment.spaceAround,
                      children: [
                        Text("cluster A:"),
                        Obx(
                          () => MenuButton<String>(
                            child: Container(
                              width: 120,
                              height: 30,
                              decoration: BoxDecoration(
                                color: Colors.white,
                                borderRadius: BorderRadius.circular(4),
                                border: Border.all(width: 1),
                              ),
                              alignment: Alignment.center,
                              padding:
                                  const EdgeInsets.symmetric(horizontal: 16),
                              child: Container(
                                width: 80,
                                height: 30,
                                color: Colors.white,
                                alignment: Alignment.center,
                                child: controller.clusterIdA != null
                                    ? Row(
                                        mainAxisAlignment:
                                            MainAxisAlignment.center,
                                        children: [
                                          Container(
                                            decoration: BoxDecoration(
                                              shape: BoxShape.circle,
                                              color: controller
                                                  .clusters[
                                                      controller.clusterIdA]
                                                  .color,
                                            ),
                                            width: 13,
                                            height: 13,
                                          ),
                                          const SizedBox(width: 10),
                                          Text(controller.clusterIdA)
                                        ],
                                      )
                                    : Text("select"),
                              ),
                            ),
                            items: controller.clustersIds,
                            topDivider: true,
                            scrollPhysics: AlwaysScrollableScrollPhysics(),
                            onItemSelected: (value) {
                              controller.clusterIdA = value;
                              if (controller.clusterIdA != null &&
                                  controller.clusterIdB != null) {
                                Get.find<ProjectionViewUiController>()
                                    .orderSeriesByRank();
                              }
                            },
                            onMenuButtonToggle: (isToggle) {},
                            decoration: BoxDecoration(
                              border:
                                  Border.all(color: Colors.white.withAlpha(0)),
                              borderRadius:
                                  const BorderRadius.all(Radius.circular(3.0)),
                              color: Colors.white.withAlpha(0),
                            ),
                            divider: Container(
                              height: 1,
                              color: Colors.grey,
                            ),
                            toggledChild: Container(
                              height: 30,
                            ),
                            itemBuilder: (String value) => Container(
                              width: 80,
                              height: 30,
                              color: Colors.white,
                              alignment: Alignment.center,
                              child: Row(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  Container(
                                    decoration: BoxDecoration(
                                      shape: BoxShape.circle,
                                      color: controller.clusters[value].color,
                                    ),
                                    width: 13,
                                    height: 13,
                                  ),
                                  SizedBox(width: 10),
                                  Text(controller.clusters[value].id)
                                ],
                              ),
                            ),
                          ),
                        ),
                        Text("cluster B:"),
                        Obx(
                          () => MenuButton<String>(
                            child: Container(
                              width: 120,
                              height: 30,
                              decoration: BoxDecoration(
                                color: Colors.white,
                                borderRadius: BorderRadius.circular(4),
                                border: Border.all(width: 1),
                              ),
                              alignment: Alignment.center,
                              padding:
                                  const EdgeInsets.symmetric(horizontal: 16),
                              child: Container(
                                width: 80,
                                height: 30,
                                color: Colors.white,
                                alignment: Alignment.center,
                                child: controller.clusterIdB != null
                                    ? Row(
                                        mainAxisAlignment:
                                            MainAxisAlignment.center,
                                        children: [
                                          Container(
                                            decoration: BoxDecoration(
                                              shape: BoxShape.circle,
                                              color: controller
                                                  .clusters[
                                                      controller.clusterIdB]
                                                  .color,
                                            ),
                                            width: 13,
                                            height: 13,
                                          ),
                                          SizedBox(width: 10),
                                          Text(controller.clusterIdB)
                                        ],
                                      )
                                    : Text("select"),
                              ),
                            ),
                            items: controller.clustersIds,
                            topDivider: true,
                            scrollPhysics: AlwaysScrollableScrollPhysics(),
                            onItemSelected: (value) {
                              controller.clusterIdB = value;
                              if (controller.clusterIdA != null &&
                                  controller.clusterIdB != null) {
                                Get.find<ProjectionViewUiController>()
                                    .orderSeriesByRank();
                              }
                            },
                            onMenuButtonToggle: (isToggle) {},
                            decoration: BoxDecoration(
                              border:
                                  Border.all(color: Colors.white.withAlpha(0)),
                              borderRadius:
                                  const BorderRadius.all(Radius.circular(3.0)),
                              color: Colors.white.withAlpha(0),
                            ),
                            divider: Container(
                              height: 1,
                              color: Colors.grey,
                            ),
                            toggledChild: Container(
                              height: 30,
                            ),
                            itemBuilder: (String value) => Container(
                              width: 80,
                              height: 30,
                              color: Colors.white,
                              alignment: Alignment.center,
                              child: Row(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  Container(
                                    decoration: BoxDecoration(
                                      shape: BoxShape.circle,
                                      color: controller.clusters[value].color,
                                    ),
                                    width: 13,
                                    height: 13,
                                  ),
                                  const SizedBox(width: 10),
                                  Text(controller.clusters[value].id)
                                ],
                              ),
                            ),
                          ),
                        ),
                      ],
                    );
                  }),
                )
              : const Center(
                  child: Text(
                  "Create two clusters to compare",
                  style: TextStyle(
                    fontSize: 14,
                    color: pColorGray,
                    fontWeight: FontWeight.w400,
                  ),
                )),
        ),
      ],
    );
  }
}
