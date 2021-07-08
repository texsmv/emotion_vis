import 'package:emotion_vis_client/app_constants.dart';
import 'package:emotion_vis_client/controllers/series_controller.dart';
import 'package:emotion_vis_client/interfaces/constants/colors.dart';
import 'package:emotion_vis_client/interfaces/general_widgets/buttons/pfilled_button.dart';
import 'package:emotion_vis_client/interfaces/general_widgets/containers/loading_container.dart';
import 'package:emotion_vis_client/interfaces/general_widgets/side_bar/side_bar.dart';
import 'package:emotion_vis_client/interfaces/modules/home/home_ui_controller.dart';
import 'package:emotion_vis_client/interfaces/modules/projection_view/projection_view.dart';
import 'package:emotion_vis_client/interfaces/modules/projection_view/projection_view_ui_controller.dart';
import 'package:emotion_vis_client/interfaces/modules/visualizations_view/visualizations_view.dart';
import 'package:emotion_vis_client/interfaces/ui_utils.dart';
import 'package:emotion_vis_client/models/visualization_levels.dart';
import 'package:emotion_vis_client/routes/route_names.dart';
import 'package:flutter/material.dart';
import 'package:flutter_icons/flutter_icons.dart';
import 'package:flutter_xlider/flutter_xlider.dart';
import 'package:get/get.dart';

class HomeUi extends GetView<HomeUiController> {
  const HomeUi({Key key}) : super(key: key);

  ProjectionViewUiController get _projectionViewController =>
      Get.find<ProjectionViewUiController>();

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Scaffold(
        // backgroundColor: const Color.fromRGBO(243, 245, 244, 1),
        body: Container(
          height: double.infinity,
          width: double.infinity,
          decoration: const BoxDecoration(
            image: DecorationImage(
              image: AssetImage(ASSET_WALLPAPER),
              fit: BoxFit.cover,
            ),
          ),
          child: Row(
            children: [
              GetBuilder<SeriesController>(
                builder: (_) => GetBuilder<HomeUiController>(
                  builder: (_) => SideBar(
                    selectedTab: controller.stackIndex,
                    tabs: [
                      ViewTab(
                        icon: Entypo.list,
                        onTap: () {
                          controller.stackIndex = 0;
                        },
                        text: "Search view",
                        options: _listViewOptions(),
                        // option:
                      ),
                      ViewTab(
                        icon: AntDesign.dotchart,
                        onTap: () {
                          controller.stackIndex = 1;
                        },
                        text: "Projection",
                        options: _projectionOptions(),
                      ),
                    ],
                    actions: [
                      ActionButton(
                        icon: FontAwesome.home,
                        onTap: () {
                          Get.find<SeriesController>().unselectDataset();
                          Get.offAllNamed(routePrincipalMenu);
                        },
                      )
                    ],
                  ),
                ),
              ),

              // * Main view
              Expanded(
                child: Container(
                  color: pColorScaffold,
                  child: GetBuilder<SeriesController>(
                    builder: (_) => GetBuilder<HomeUiController>(
                      builder: (_) => Column(
                        children: [
                          Expanded(
                            child: IndexedStack(
                              index: controller.stackIndex,
                              // ignore: prefer_const_literals_to_create_immutables
                              children: [
                                VisualizationsView(),
                                ProjectionView(),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  List<Widget> _listViewOptions() {
    final List<Widget> options = [];

    // * Adding visualizations chart options
    if (controller.selectedVisualization == MultipleView.LIST) {
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
    }

    // * Adding dateformat options
    if (controller.datasetSettings.isDated) {
      options.add(
        Column(
          crossAxisAlignment: CrossAxisAlignment.center,
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

  List<Widget> _projectionOptions() {
    final List<Widget> options = [];

    // * Adding projection algorithm selection
    final List<int> projections = [0, 1, 2, 3];
    options.add(
      Obx(
        () => LoadingContainer(
          isLoading: controller.datasetSettings.updating,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Padding(
                padding: EdgeInsets.symmetric(vertical: 10, horizontal: 5),
                child: Text(
                  "Dimensionality reduction:".toUpperCase(),
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
                itemCount: projections.length,
                shrinkWrap: true,
                itemBuilder: (context, index) {
                  return InkWell(
                    onTap: () {
                      controller.changeProjection(projections[index]);
                      controller.update();
                    },
                    child: Column(
                      children: [
                        Container(
                          height: 40,
                          decoration: BoxDecoration(
                            color: controller.datasetSettings.projection ==
                                    projections[index]
                                ? pColorAccent.withOpacity(0.7)
                                : const Color.fromARGB(0, 240, 240, 240),
                          ),
                          alignment: Alignment.centerLeft,
                          padding: const EdgeInsets.symmetric(horizontal: 10),
                          child: Text(
                            uiUtilProjectionToStr(
                              projections[index],
                            ),
                            textAlign: TextAlign.start,
                            overflow: TextOverflow.ellipsis,
                            style: TextStyle(
                              color: controller.datasetSettings.projection ==
                                      projections[index]
                                  ? Colors.white
                                  : Colors.black,
                            ),
                          ),
                        ),
                        if (projections[index] != 0)
                          AnimatedContainer(
                            color: Colors.white.withOpacity(0.6),
                            duration: const Duration(milliseconds: 300),
                            height: controller.datasetSettings.projection ==
                                    projections[index]
                                ? 85
                                : 0,
                            child: SingleChildScrollView(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  const SizedBox(height: 10),
                                  Padding(
                                    padding: const EdgeInsets.symmetric(
                                        horizontal: 10),
                                    child: Text(
                                      "${uiUtilProjectionParamStr(
                                        projections[index],
                                      )}:",
                                    ),
                                  ),
                                  FlutterSlider(
                                    handlerHeight: 25,
                                    handlerWidth: 25,
                                    handler: FlutterSliderHandler(
                                      child: Container(),
                                    ),
                                    hatchMark: FlutterSliderHatchMark(
                                      labelsDistanceFromTrackBar: 33,

                                      density: 2,
                                      //     0.5, // means 50 lines, from 0 to 100 percent
                                      labels: [
                                        FlutterSliderHatchMarkLabel(
                                          percent: 0,
                                          label: Text(
                                            uiUtilProjectionParamRange(
                                                    projections[index])
                                                .item1
                                                .toString(),
                                          ),
                                        ),
                                        FlutterSliderHatchMarkLabel(
                                          percent: 100,
                                          label: Text(
                                              uiUtilProjectionParamRange(
                                                      projections[index])
                                                  .item2
                                                  .toString()),
                                        ),
                                      ],
                                    ),
                                    values: [
                                      controller.datasetSettings
                                          .getProjectionParameter(
                                              projections[index])
                                          .toDouble()
                                    ],
                                    min: uiUtilProjectionParamRange(
                                            projections[index])
                                        .item1
                                        .toDouble(),
                                    max: uiUtilProjectionParamRange(
                                            projections[index])
                                        .item2
                                        .toDouble(),
                                    onDragCompleted: (handlerIndex, lowerValue,
                                        upperValuealue) {
                                      controller.changeProjectionParameter(
                                        lowerValue.toInt(),
                                      );
                                      controller.update();
                                    },
                                  ),
                                ],
                              ),
                            ),
                          )
                        else
                          Container(),
                      ],
                    ),
                  );
                },
              ),
            ],
          ),
        ),
      ),
    );

    // * Adding projection algorithm selection
    final List<ClusteringMethod> clusteringMethods =
        ClusteringMethod.values.reversed.toList();
    options.add(
      GetBuilder<ProjectionViewUiController>(
        builder: (_) => Obx(
          () => LoadingContainer(
            isLoading: controller.datasetSettings.updating,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                Padding(
                  padding: EdgeInsets.symmetric(vertical: 10, horizontal: 5),
                  child: Text(
                    "Clustering:".toUpperCase(),
                    style: TextStyle(
                      fontWeight: FontWeight.w600,
                      fontSize: 15,
                    ),
                    textAlign: TextAlign.center,
                  ),
                ),
                ListView.separated(
                  physics: const NeverScrollableScrollPhysics(),
                  separatorBuilder: (context, index) =>
                      const SizedBox(height: 5),
                  itemCount: ClusteringMethod.values.length,
                  shrinkWrap: true,
                  itemBuilder: (context, index) {
                    return Column(
                      children: [
                        InkWell(
                          onTap: () {
                            _projectionViewController.changeClusteringMethod(
                                clusteringMethods[index]);
                          },
                          child: Container(
                            height: 40,
                            decoration: BoxDecoration(
                              color:
                                  _projectionViewController.clusteringMethod ==
                                          clusteringMethods[index]
                                      ? pColorAccent.withOpacity(0.7)
                                      : const Color.fromARGB(0, 240, 240, 240),
                            ),
                            alignment: Alignment.centerLeft,
                            padding: const EdgeInsets.symmetric(horizontal: 10),
                            child: Text(
                              uiUtilClustering2Str(
                                clusteringMethods[index],
                              ),
                              textAlign: TextAlign.start,
                              overflow: TextOverflow.ellipsis,
                              style: TextStyle(
                                color: _projectionViewController
                                            .clusteringMethod ==
                                        clusteringMethods[index]
                                    ? Colors.white
                                    : Colors.black,
                              ),
                            ),
                          ),
                        ),
                        if (clusteringMethods[index] ==
                                ClusteringMethod.dbscan ||
                            clusteringMethods[index] == ClusteringMethod.kmeans)
                          AnimatedContainer(
                            color: Colors.white.withOpacity(0.6),
                            duration: const Duration(milliseconds: 300),
                            height: (_projectionViewController
                                        .clusteringMethod ==
                                    clusteringMethods[index])
                                ? _projectionViewController.clusteringMethod ==
                                        ClusteringMethod.kmeans
                                    ? 85
                                    : 125
                                : 0,
                            child: SingleChildScrollView(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  const SizedBox(height: 10),
                                  Padding(
                                    padding: const EdgeInsets.symmetric(
                                        horizontal: 10),
                                    child: ClusteringMethod.kmeans ==
                                            clusteringMethods[index]
                                        ? Column(
                                            children: [
                                              Row(
                                                mainAxisAlignment:
                                                    MainAxisAlignment
                                                        .spaceBetween,
                                                children: [
                                                  const Text(
                                                    "k:",
                                                  ),
                                                  Container(
                                                    height: 30,
                                                    width: 50,
                                                    padding: const EdgeInsets
                                                            .symmetric(
                                                        horizontal: 3),
                                                    decoration: BoxDecoration(
                                                      borderRadius:
                                                          BorderRadius.circular(
                                                              5),
                                                      color: Colors.white,
                                                    ),
                                                    child: TextField(
                                                      controller:
                                                          _projectionViewController
                                                              .kController,
                                                    ),
                                                  )
                                                ],
                                              ),
                                              PFilledButton(
                                                text: "run",
                                                onPressed: () {
                                                  _projectionViewController
                                                      .changeClusteringMethod(
                                                          clusteringMethods[
                                                              index]);
                                                },
                                              )
                                            ],
                                          )
                                        // * dbscan case
                                        : Column(
                                            children: [
                                              Row(
                                                mainAxisAlignment:
                                                    MainAxisAlignment
                                                        .spaceBetween,
                                                children: [
                                                  const Text(
                                                    "eps:",
                                                  ),
                                                  Container(
                                                    height: 30,
                                                    width: 50,
                                                    padding: const EdgeInsets
                                                            .symmetric(
                                                        horizontal: 3),
                                                    decoration: BoxDecoration(
                                                      borderRadius:
                                                          BorderRadius.circular(
                                                              5),
                                                      color: Colors.white,
                                                    ),
                                                    child: TextField(
                                                      controller:
                                                          _projectionViewController
                                                              .epsController,
                                                    ),
                                                  )
                                                ],
                                              ),
                                              const SizedBox(height: 5),
                                              Row(
                                                mainAxisAlignment:
                                                    MainAxisAlignment
                                                        .spaceBetween,
                                                children: [
                                                  Text(
                                                    "n_samples:",
                                                  ),
                                                  Container(
                                                    height: 30,
                                                    width: 50,
                                                    padding: const EdgeInsets
                                                            .symmetric(
                                                        horizontal: 3),
                                                    decoration: BoxDecoration(
                                                      borderRadius:
                                                          BorderRadius.circular(
                                                              5),
                                                      color: Colors.white,
                                                    ),
                                                    child: TextField(
                                                      controller:
                                                          _projectionViewController
                                                              .nsamplesController,
                                                    ),
                                                  )
                                                ],
                                              ),
                                              PFilledButton(
                                                text: "run",
                                                onPressed: () {
                                                  _projectionViewController
                                                      .changeClusteringMethod(
                                                          clusteringMethods[
                                                              index]);
                                                },
                                              )
                                            ],
                                          ),
                                  ),
                                ],
                              ),
                            ),
                          )
                        else
                          Container(),
                      ],
                    );
                  },
                ),
              ],
            ),
          ),
        ),
      ),
    );
    options.add(
      GetBuilder<ProjectionViewUiController>(
        builder: (_) => InkWell(
          onTap: () {
            if (_projectionViewController.allowSelection) {
              _projectionViewController.allowSelection = false;
            } else {
              _projectionViewController.allowSelection = true;
            }
            _projectionViewController.update();
          },
          child: Container(
            height: 40,
            alignment: Alignment.center,
            child: _projectionViewController.allowSelection
                ? Text(
                    "Cancelar",
                    style: TextStyle(
                      color: Colors.red.shade700,
                      fontWeight: FontWeight.w500,
                    ),
                  )
                : Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const Icon(
                        Icons.add,
                        color: pColorDark,
                      ),
                      const Text(
                        "Add cluster",
                        style: TextStyle(
                          color: pColorDark,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                      SizedBox(width: 5),
                    ],
                  ),
          ),
        ),
      ),
    );

    options.add(
      InkWell(
        onTap: () {
          _projectionViewController
              .changeClusteringMethod(ClusteringMethod.none);
        },
        child: Container(
          height: 40,
          alignment: Alignment.center,
          child: Text(
            "Clear clusters",
            style: TextStyle(
              color: Colors.red.shade700,
              fontWeight: FontWeight.w500,
            ),
          ),
        ),
      ),
    );

    // * Adding distance selection
    final List<int> distances = [0, 1];
    options.add(
      Obx(
        () => LoadingContainer(
          isLoading: controller.datasetSettings.updating,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              Padding(
                padding: EdgeInsets.symmetric(vertical: 10, horizontal: 5),
                child: Text(
                  "Distance:".toUpperCase(),
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
                itemCount: distances.length,
                shrinkWrap: true,
                itemBuilder: (context, index) {
                  return InkWell(
                    onTap: () {
                      controller.changeDistance(distances[index]);
                      controller.update();
                    },
                    child: Container(
                      height: 40,
                      decoration: BoxDecoration(
                        color: controller.datasetSettings.distance ==
                                distances[index]
                            ? pColorAccent.withOpacity(0.7)
                            : const Color.fromARGB(0, 240, 240, 240),
                      ),
                      alignment: Alignment.centerLeft,
                      padding: const EdgeInsets.symmetric(horizontal: 10),
                      child: Text(
                        uiUtilDistanceToStr(
                          distances[index],
                        ),
                        textAlign: TextAlign.start,
                        overflow: TextOverflow.ellipsis,
                        style: TextStyle(
                          color: controller.datasetSettings.distance ==
                                  distances[index]
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
        ),
      ),
    );

    return options;
  }
}
