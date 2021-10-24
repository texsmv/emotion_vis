import 'dart:math';

import 'package:emotion_vis_client/controllers/series_controller.dart';
import 'package:emotion_vis_client/interfaces/ui_utils.dart';
import 'package:emotion_vis_client/models/cluster_model.dart';
import 'package:emotion_vis_client/models/dataset_settings.dart';
import 'package:emotion_vis_client/models/person_model.dart';
import 'package:emotion_vis_client/repositories/series_repository.dart';
import 'package:flutter/cupertino.dart';
import 'package:get/get.dart';
import 'package:tuple/tuple.dart';
import 'package:emotion_vis_client/list_extension.dart';

// enum ClusteringMethod { automatic, manual, byLabel, none }
enum ClusteringMethod { kmeans, dbscan, categorical, none }

class ProjectionViewUiController extends GetxController {
  SeriesController _seriesController = Get.find();
  DatasetSettings get datasetSettings => _seriesController.datasetSettings;

  String get clusterIdA => _seriesController.clusterIdA;
  String get clusterIdB => _seriesController.clusterIdB;
  set clusterIdA(String value) {
    _seriesController.clusterIdA = null;
    _seriesController.clusterIdA = value;
    repositoryGetTemporalGroupSummary(
            _seriesController.selectedDatasetId,
            datasetSettings.begin,
            datasetSettings.end,
            _seriesController.clusters[clusterIdA].personsIds)
        .then((data) {
      final List<String> keys = data.keys.toList();
      clusterATemporalStats = {};
      for (int i = 0; i < keys.length; i++) {
        clusterATemporalStats[keys[i]] =
            Map<String, List<dynamic>>.from(data[keys[i]]);
      }
      clusterATemporalStats = data;
      update();
    });
    repositoryGetInstanceGroupSummary(
            _seriesController.selectedDatasetId,
            datasetSettings.begin,
            datasetSettings.end,
            _seriesController.clusters[clusterIdA].personsIds)
        .then((data) {
      final List<String> keys = data.keys.toList();
      clusterAInstantStats = {};
      for (int i = 0; i < keys.length; i++) {
        clusterAInstantStats[keys[i]] = Map<String, double>.from(data[keys[i]]);
      }
      update();
    });

    if (datasetSettings.modelType == ModelType.DIMENSIONAL) {
      repositoryGetHistogram(
              _seriesController.selectedDatasetId,
              datasetSettings.begin,
              datasetSettings.end,
              _seriesController.clusters[clusterIdA].personsIds,
              datasetSettings.arousal,
              datasetSettings.valence)
          .then((data) {
        List<dynamic> hist = List.from(data["histogram"]).reshape([20, 20]);
        histogramA = List<List<int>>.generate(
            hist.shape[0],
            (index) => List.generate(
                hist.shape[1], (index2) => hist[index][index2].toInt()));
        histogramAmaxCount = data["cellCount"].toInt();
        update();
      });
    }
  }

  set clusterIdB(String value) {
    _seriesController.clusterIdB = null;
    _seriesController.clusterIdB = value;
    repositoryGetTemporalGroupSummary(
            _seriesController.selectedDatasetId,
            datasetSettings.begin,
            datasetSettings.end,
            _seriesController.clusters[clusterIdB].personsIds)
        .then((data) {
      final List<String> keys = data.keys.toList();
      clusterBTemporalStats = {};
      for (int i = 0; i < keys.length; i++) {
        clusterBTemporalStats[keys[i]] =
            Map<String, List<dynamic>>.from(data[keys[i]]);
      }
      clusterBTemporalStats = data;
      update();
    });
    repositoryGetInstanceGroupSummary(
            _seriesController.selectedDatasetId,
            datasetSettings.begin,
            datasetSettings.end,
            _seriesController.clusters[clusterIdB].personsIds)
        .then((data) {
      final List<String> keys = data.keys.toList();
      clusterBInstantStats = {};
      for (int i = 0; i < keys.length; i++) {
        clusterBInstantStats[keys[i]] = Map<String, double>.from(data[keys[i]]);
      }
      update();
    });
    if (datasetSettings.modelType == ModelType.DIMENSIONAL) {
      repositoryGetHistogram(
              _seriesController.selectedDatasetId,
              datasetSettings.begin,
              datasetSettings.end,
              _seriesController.clusters[clusterIdB].personsIds,
              datasetSettings.arousal,
              datasetSettings.valence)
          .then((data) {
        List<dynamic> hist = List.from(data["histogram"]).reshape([20, 20]);
        histogramB = List<List<int>>.generate(
            hist.shape[0],
            (index) => List.generate(
                hist.shape[1], (index2) => hist[index][index2].toInt()));
        histogramBmaxCount = data["cellCount"].toInt();
        update();
      });
    }
  }

  Map<String, dynamic> clusterATemporalStats;
  Map<String, dynamic> clusterBTemporalStats;
  Map<String, Map<String, dynamic>> clusterAInstantStats;
  Map<String, Map<String, dynamic>> clusterBInstantStats;
  bool get areStatsLoaded {
    if (datasetSettings.modelType == ModelType.DISCRETE) {
      if (clusterATemporalStats == null ||
          clusterBTemporalStats == null ||
          clusterAInstantStats == null ||
          clusterBInstantStats == null) return false;
    } else {
      if (clusterATemporalStats == null ||
          clusterBTemporalStats == null ||
          clusterAInstantStats == null ||
          clusterBInstantStats == null ||
          histogramA == null ||
          histogramB == null ||
          histogramAmaxCount == null ||
          histogramBmaxCount == null) {
        return false;
      }
    }
    return true;
  }

  List<List<int>> histogramA;
  List<List<int>> histogramB;
  int histogramAmaxCount;
  int histogramBmaxCount;

  TextEditingController kController;
  TextEditingController epsController;
  TextEditingController nsamplesController;

  Random rng = Random();

  List<InteractivePoint> points;

  double windowWidth = 100;
  double windowHeigth = 100;
  RxDouble infoHeight = 40.0.obs;
  RxDouble infoXposition = 0.0.obs;
  RxDouble infoYposition = 0.0.obs;
  RxBool showInfo = false.obs;
  InteractivePoint _hoveredPoint;
  InteractivePoint get hoveredPoint => _hoveredPoint;
  set hoveredPoint(InteractivePoint value) {
    _hoveredPoint = value;
    // for the id
    double height = 50;
    if (datasetSettings.categoricalData.isNotEmpty) {
      height += 20 * datasetSettings.categoricalData.length;
    }
    if (datasetSettings.numericalLabels.isNotEmpty) {
      height += 20 * datasetSettings.numericalLabels.length;
    }
    infoHeight.value = height;
  }

  String get personId {
    return _hoveredPoint.personModel.id;
  }

  List<PersonModel> get blueCluster => _seriesController.blueCluster;
  List<PersonModel> get redCluster => _seriesController.redCluster;
  List<String> get variablesOrdered =>
      _seriesController.variablesNamesOrdered ?? datasetSettings.variablesNames;

  List<String> get clustersIds => _seriesController.clustersIds;
  Map<String, ClusterModel> get clusters => _seriesController.clusters;

  double plotMargin = 45;

  RxBool _allowSelection = false.obs;
  bool get allowSelection => _allowSelection.value;
  set allowSelection(bool value) => _allowSelection.value = value;

  Rx<Offset> _selectionBeginPosition = Offset(0, 0).obs;
  Offset get selectionBeginPosition => _selectionBeginPosition.value;
  set selectionBeginPosition(Offset value) =>
      _selectionBeginPosition.value = value;
  Rx<Offset> _selectionEndPosition = Offset(0, 0).obs;
  Offset get selectionEndPosition => _selectionEndPosition.value;
  set selectionEndPosition(Offset value) => _selectionEndPosition.value = value;

  RxBool _flipHorizontally = false.obs;
  bool get flipHorizontally => _flipHorizontally.value;
  set flipHorizontally(bool value) => _flipHorizontally.value = value;
  RxBool _flipVertically = false.obs;
  bool get flipVertically => _flipVertically.value;
  set flipVertically(bool value) => _flipVertically.value = value;

  ClusteringMethod clusteringMethod = ClusteringMethod.none;

  @override
  void onInit() {
    kController = TextEditingController(text: "3");
    epsController = TextEditingController(text: "0.2");
    nsamplesController = TextEditingController(text: "5");
    createPoints();
    uiUtilDelayed(() {
      update();
    }, delay: const Duration(milliseconds: 500));
    super.onInit();
  }

  double get selectionWidth =>
      (selectionEndPosition.dx - selectionBeginPosition.dx).abs();

  double get selectionHeight =>
      (selectionEndPosition.dy - selectionBeginPosition.dy).abs();

  double get selectionHorizontalStart {
    if (flipHorizontally) {
      return selectionBeginPosition.dx - selectionWidth;
    } else {
      return selectionBeginPosition.dx;
    }
  }

  double get selectionVerticalStart {
    if (flipVertically) {
      return selectionBeginPosition.dy - selectionHeight;
    } else {
      return selectionBeginPosition.dy;
    }
  }

  void onPointerDown(PointerDownEvent event) {
    if (allowSelection) {
      selectionBeginPosition = event.localPosition;
    }
  }

  void onPointerMove(PointerMoveEvent event) {
    if (allowSelection) {
      selectionEndPosition = event.localPosition;
      if ((selectionEndPosition.dx - selectionBeginPosition.dx).isNegative) {
        flipHorizontally = true;
      } else {
        flipHorizontally = false;
      }
      if ((selectionEndPosition.dy - selectionBeginPosition.dy).isNegative) {
        flipVertically = true;
      } else {
        flipVertically = false;
      }
    }
  }

  void manuallyAddCluster() {
    final List<InteractivePoint> selectedPoints = getSelectedPoints();
    _seriesController.addCustomCluster(selectedPoints);
  }

  void onPointerUp(PointerUpEvent event) {
    if (allowSelection) {
      manuallyAddCluster();
      selectionBeginPosition = Offset(0, 0);
      selectionEndPosition = Offset(0, 0);
      allowSelection = false;
    }
  }

  void addNewCluster() {
    allowSelection = true;
  }

  double mapToAxisX(double value) {
    if (value.abs() > 1) {
      print("value excedding: $value");
    }
    return uiUtilRangeConverter(
        value, -1, 1, 0 + plotMargin, windowWidth - plotMargin);
  }

  List<InteractivePoint> getSelectedPoints() {
    List<InteractivePoint> selected = [];
    for (var i = 0; i < points.length; i++) {
      double x = points[i].plotCoordinates.item1;
      double y = points[i].plotCoordinates.item2;
      if ((x >
              min(selectionHorizontalStart,
                  selectionHorizontalStart + selectionWidth)) &&
          (x <
              max(selectionHorizontalStart,
                  selectionHorizontalStart + selectionWidth)) &&
          (y >
              min(selectionVerticalStart,
                  selectionVerticalStart + selectionHeight)) &&
          (y <
              max(selectionVerticalStart,
                  selectionVerticalStart + selectionHeight))) {
        selected.add(points[i]);
      }
    }
    return selected;
  }

  double mapToAxisY(double value) {
    return uiUtilRangeConverter(
        value, -1, 1, 0 + plotMargin, windowHeigth - plotMargin);
  }

  Future<void> changeClusteringMethod(
    ClusteringMethod method,
  ) async {
    _seriesController.removeClusters();
    if (method == ClusteringMethod.none) {
      _seriesController.removeClusters();
    } else if (method == ClusteringMethod.kmeans) {
      final int k = int.tryParse(kController.text) ?? null;
      if (k == null) {
        uiUtilDialog(Container(
          padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 20),
          alignment: Alignment.center,
          child: const Text(
            "k must be an integer.",
            textAlign: TextAlign.center,
          ),
        ));
        return;
      }
      _seriesController.kmeansClustering(
        k: k,
      );
    } else if (method == ClusteringMethod.dbscan) {
      final int min_samples = int.tryParse(nsamplesController.text) ?? null;
      if (min_samples == null) {
        uiUtilDialog(Container(
          padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 20),
          alignment: Alignment.center,
          child: const Text(
            "min_samples must be an integer.",
            textAlign: TextAlign.center,
          ),
        ));
        return;
      }

      final double eps = double.tryParse(epsController.text) ?? null;
      if (eps == null) {
        uiUtilDialog(Container(
          padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 20),
          alignment: Alignment.center,
          child: const Text(
            "eps must be a double.",
            textAlign: TextAlign.center,
          ),
        ));
        return;
      }
      _seriesController.dbscanClustering(
        min_samples: min_samples,
        eps: eps,
      );
    } else if (method == ClusteringMethod.categorical) {
      String filter = await showFilterOptions();
      if (filter != null) {
        _seriesController.labeledClustering(filter);
      } else {
        clusteringMethod = ClusteringMethod.none;
        update();
        return;
      }
    }
    clusteringMethod = method;
    update();
  }

  Future<String> showFilterOptions() async {
    return await uiUtilDialog(Container(
      height: 300,
      width: 200,
      child: datasetSettings.categoricalLabels.length != 0
          ? Column(
              children: [
                const SizedBox(height: 10),
                const Text(
                  "Cluster by",
                  style: TextStyle(
                    fontWeight: FontWeight.w600,
                    fontSize: 16,
                  ),
                ),
                Expanded(
                  child: ListView.builder(
                    itemBuilder: (context, index) {
                      return GestureDetector(
                        onTap: () {
                          Get.back(
                              result: datasetSettings.categoricalLabels[index]);
                        },
                        behavior: HitTestBehavior.opaque,
                        child: Container(
                          height: 40,
                          alignment: Alignment.center,
                          child: Text(
                            datasetSettings.categoricalLabels[index],
                            textAlign: TextAlign.center,
                          ),
                        ),
                      );
                    },
                    itemCount: datasetSettings.categoricalLabels.length,
                  ),
                )
              ],
            )
          : Center(child: Text("No labels found")),
    ));
  }

  void createPoints() {
    points = [];
    for (var i = 0; i < _seriesController.persons.length; i++) {
      points.add(
        InteractivePoint(
          personModel: _seriesController.persons[i],
          // coordinates: Tuple2(uiUtilRandomDoubleInRange(rng, 5, 15),
          // uiUtilRandomDoubleInRange(rng, 5, 15)),
        ),
      );
    }
    projectPointsToPlot();
  }

  void updatePoints() {
    for (var i = 0; i < _seriesController.persons.length; i++) {
      points[i].personModel = _seriesController.persons[i];
      // points.add(
      //   InteractivePoint(
      //     personModel: _seriesController.persons[i],
      //     // coordinates: Tuple2(uiUtilRandomDoubleInRange(rng, 5, 15),
      //     // uiUtilRandomDoubleInRange(rng, 5, 15)),
      //   ),
      // );
    }
    projectPointsToPlot();
  }

  void projectPointsToPlot() {
    for (var i = 0; i < points.length; i++) {
      points[i].plotCoordinates = Tuple2(
        mapToAxisX(points[i].coordinates.item1),
        mapToAxisY(points[i].coordinates.item2),
      );
      // print({"i: $i - ${points[i].plotCoordinates}"});
    }
  }

  void updateProjections() async {
    await _seriesController.getDatasetProjection();
    projectPointsToPlot();
    update();
  }

  void orderSeriesByRank() async {
    await _seriesController.getFishersDiscriminantRanking(
        clusterIdA, clusterIdB);
  }
}

class InteractivePoint {
  PersonModel personModel;
  Tuple2<double, double> get coordinates =>
      Tuple2(personModel.x, personModel.y);
  Tuple2<double, double> plotCoordinates;

  InteractivePoint({this.personModel});
}
