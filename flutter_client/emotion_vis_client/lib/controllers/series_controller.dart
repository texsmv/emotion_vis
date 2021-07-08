import 'package:emotion_vis_client/enums/app_enums.dart';
import 'package:emotion_vis_client/interfaces/modules/projection_view/projection_view_ui_controller.dart';
import 'package:emotion_vis_client/interfaces/ui_utils.dart';
import 'package:emotion_vis_client/models/MTSerie.dart';
import 'package:emotion_vis_client/models/cluster_model.dart';
import 'package:emotion_vis_client/models/dataset_settings.dart';
import 'package:emotion_vis_client/models/person_model.dart';
import 'package:emotion_vis_client/models/visualization_levels.dart';
import 'package:emotion_vis_client/repositories/series_repository.dart';
import 'package:emotion_vis_client/utils.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';

class SeriesController extends GetxController {
  List<String> loadedDatasetsIds = [];
  List<String> localDatasetsIds = [];
  List<PersonModel> _persons;
  RxString _clusterIdA = RxString();
  RxString _clusterIdB = RxString();
  String get clusterIdA => _clusterIdA.value;
  String get clusterIdB => _clusterIdB.value;
  set clusterIdA(String value) {
    _clusterIdA.value = value;
    if (value == null) {
      Get.find<ProjectionViewUiController>().clusterAInstantStats = null;
      Get.find<ProjectionViewUiController>().clusterATemporalStats = null;
      Get.find<ProjectionViewUiController>().histogramA = null;
      Get.find<ProjectionViewUiController>().histogramAmaxCount = null;
    }
  }

  set clusterIdB(String value) {
    _clusterIdB.value = value;
    if (value == null) {
      Get.find<ProjectionViewUiController>().clusterBInstantStats = null;
      Get.find<ProjectionViewUiController>().clusterBTemporalStats = null;
      Get.find<ProjectionViewUiController>().histogramB = null;
      Get.find<ProjectionViewUiController>().histogramBmaxCount = null;
    }
  }

  List<PersonModel> blueCluster = [];
  List<PersonModel> redCluster = [];

  Map<String, ClusterModel> clusters = {};
  List<String> get clustersIds => clusters.keys.toList();
  List<String> variablesNamesOrdered;
  List<PersonModel> get persons => _persons;
  String selectedDatasetId;
  DatasetSettings datasetSettings;
  Map<OverviewType, Map<String, List<double>>> overview;
  // ignore: non_constant_identifier_names
  Map<String, dynamic> d_k;
  Map<String, dynamic> coords;
  bool visualizeAllTime = true;

  Future<void> updateSettings({
    int windowPosition,
    int windowLength,
    Map<String, double> alphas,
    Map<String, Color> colors,
    ModelType modelType,
    String valence,
    String dominance,
    String arousal,
    DateStrFormat dateFormat,
    int distance,
    int projection,
    int projectionParameter,
  }) async {
    print("----DONE UPDATING---");
    if (windowLength != null ||
        windowPosition != null ||
        distance != null ||
        projection != null ||
        projectionParameter != null) {
      datasetSettings.updating = true;
      if (distance != null) datasetSettings.distance = distance;
      if (projection != null) {
        datasetSettings.projection = projection;
      }
      if (projectionParameter != null) {
        datasetSettings.projectionParameter = projectionParameter;
      }
      if (windowLength != null) datasetSettings.windowLength = windowLength;
      if (windowPosition != null) {
        datasetSettings.windowPosition = windowPosition;
      }
      d_k = null;
      coords = null;
      print("GETTING MTSERIES");
      await getMTSeries(
        datasetSettings.ids,
      );
      print("GETTING PROYECTION");
      await getDatasetProjection();
      datasetSettings.updating = false;
    }
    if (alphas != null) datasetSettings.alphas = alphas;
    if (colors != null) datasetSettings.variablesColors = colors;
    if (modelType != null) datasetSettings.modelType = modelType;
    if (valence != null) datasetSettings.valence = valence;
    if (dominance != null) datasetSettings.dominance = dominance;
    if (arousal != null) datasetSettings.valence = arousal;
    if (dateFormat != null) {
      datasetSettings.dateFormat = dateFormat;
      datasetSettings.formatDateTimes();
    }
    print("----DONE UPDATE---");
    update();
  }

  Future<NotifierState> downsampleData(DownsampleRule rule) async {
    await repositoryDownsampleData(
        selectedDatasetId, Utils.downsampleRule2Str(rule));
    return NotifierState.SUCCESS;
  }

  Future<NotifierState> getFishersDiscriminantRanking(
      String clusterA, String clusterB) async {
    final List<String> blueClusterIds = clusters[clusterA].personsIds;
    final List<String> redClusterIds = clusters[clusterB].personsIds;
    if (blueClusterIds.isEmpty || redClusterIds.isEmpty) {
      return NotifierState.ERROR;
    }

    variablesNamesOrdered = await repositoryGetFishersDiscriminantRanking(
      selectedDatasetId,
      d_k,
      blueClusterIds,
      redClusterIds,
    );
    blueCluster = List.generate(blueClusterIds.length,
        (index) => personModelById(blueClusterIds[index]));
    redCluster = List.generate(
        redClusterIds.length, (index) => personModelById(redClusterIds[index]));
    update();
    return NotifierState.SUCCESS;
  }

  PersonModel personModelById(String id) {
    for (var i = 0; i < persons.length; i++) {
      if (persons[i].id == id) {
        return persons[i];
      }
    }
    return PersonModel(mtSerie: MTSerie(dateTimes: [], timeSeries: {}));
  }

  Future<NotifierState> getDatasetsInfo() async {
    final Map<String, List<String>> data = await repositoryGetDatasetsInfo();
    loadedDatasetsIds = data["loadedDatasetsIds"];
    localDatasetsIds = data["localDatasetsIds"];
    update();
    return NotifierState.SUCCESS;
  }

  Future<void> getDatasetInfo() async {
    Map<String, dynamic> infoMap =
        await repositoryGetDatasetInfo(selectedDatasetId);
    datasetSettings = DatasetSettings.fromMap(info: infoMap);
    update();
  }

  Future<NotifierState> loadLocalDataset(String datasetId) async {
    final NotifierState state = await repositoryLoadLocalDataset(datasetId);
    update();
    return state;
  }

  Future<NotifierState> removeDataset(String datasetId) async {
    final NotifierState state = await repositoryRemoveDataset(datasetId);
    update();
    return state;
  }

  /// Returns "error" if an error ocurred
  Future<String> initializeDataset(String datasetInfoJson) async {
    return repositoryInitializeDataset(datasetInfoJson);
  }

  Future<NotifierState> addEml(String datasetId, String eml) async {
    return repositoryAddEml(datasetId, eml);
  }

  // Future<PersonModel> getPersonModel(String id, begin, end) async {
  //   Map<String, dynamic> queryMap = await repositoryGetMTSeries(
  //     selectedDatasetId,
  //     begin,
  //     end,
  //     [id],
  //   );
  //   Map dimensions = queryMap[id]["temporalVariables"];
  //   PersonModel personModel = PersonModel.fromMap(map: dimensions, id: id);
  //   personModel.metadata = queryMap[id]['metadata'];
  //   personModel.categoricalValues =
  //       queryMap[id]['categoricalFeatures'].cast<String>();
  //   personModel.categoricalLabels =
  //       queryMap[id]['categoricalLabels'].cast<String>();
  //   personModel.numericalValues =
  //       queryMap[id]['numericalFeatures'].cast<double>();
  //   personModel.numericalLabels =
  //       queryMap[id]['numericalLabels'].cast<String>();
  //   return personModel;
  // }

  Future<NotifierState> getMTSeries(List<String> ids) async {
    bool updateMode = _persons != null;
    print("UpdateMode: $updateMode");

    if (!updateMode) {
      _persons = [];
    }
    Map<String, dynamic> queryMap = await repositoryGetMTSeries(
      selectedDatasetId,
      datasetSettings.begin,
      datasetSettings.end,
      ids,
      getEmotions: true,
    );

    for (int i = 0; i < ids.length; i++) {
      Map dimensions = queryMap[ids[i]]["temporalVariables"];
      PersonModel personModel =
          PersonModel.fromMap(map: dimensions, id: ids[i]);
      personModel.metadata = queryMap[ids[i]]['metadata'];
      personModel.categoricalValues =
          queryMap[ids[i]]['categoricalFeatures'].cast<String>();
      personModel.categoricalLabels =
          queryMap[ids[i]]['categoricalLabels'].cast<String>();
      personModel.numericalValues =
          queryMap[ids[i]]['numericalFeatures'].cast<double>();
      personModel.numericalLabels =
          queryMap[ids[i]]['numericalLabels'].cast<String>();
      if (updateMode) {
        personModel.clusterId = _persons[i].clusterId;
        personModel.x = _persons[i].x;
        personModel.y = _persons[i].y;
        _persons[i] = personModel;
      } else {
        _persons.add(personModel);
      }
    }
    update();
    return NotifierState.SUCCESS;
  }

  /// Gets the current dataset projection
  ///
  /// Uses Euclidean if [distance] is equal to 0 and DTW if it is 1.
  /// Uses MDS if [projection] is equal to 0, ISOMAP for 1, UMAP for 2 and
  /// TSNE for 3.
  /// [projectionParameter] is the parameter used for the projection
  Future<NotifierState> getDatasetProjection() async {
    Map<String, dynamic> data = await repositoryGetDatasetProjection(
      selectedDatasetId,
      datasetSettings.begin,
      datasetSettings.end,
      datasetSettings.distance,
      datasetSettings.projection,
      datasetSettings.projectionParameter,
      datasetSettings.alphas,
      d_k: d_k ?? {},
      oldCoords: coords ?? {},
    );
    coords = data["coords"];
    d_k = data["D_k"];

    for (var i = 0; i < persons.length; i++) {
      var coord = coords[persons[i].id];
      persons[i].x = coord[0];
      persons[i].y = coord[1];
    }
    update();
    return NotifierState.SUCCESS;
  }

  /// Cluster all the persons using K-means
  ///
  /// The [id] and [color] are assigned by the order.
  /// The [id] of each cluster is set to each [PersonModel] in the
  /// cluster
  Future<NotifierState> kmeansClustering({int k = 4}) async {
    Map<String, dynamic> result =
        await repositoryKmeansClustering(selectedDatasetId, coords, k);

    final List<String> keys = result.keys.toList();
    final List<String> clusterIds = List.generate(
        keys.length, (index) => "cluster ${int.parse(keys[index])}");
    clusters = {};
    for (var i = 0; i < clusterIds.length; i++) {
      final List<String> clusterElements = List<String>.from(result[keys[i]]);
      final List<PersonModel> clusterPersons = [];
      for (var j = 0; j < clusterElements.length; j++) {
        final PersonModel personModel = personModelById(clusterElements[j]);
        personModel.clusterId = clusterIds[i];
        clusterPersons.add(personModel);
      }
      final ClusterModel newCluster = ClusterModel(
        id: clusterIds[i],
        color: uiUtilGetClusteringColor(i),
        persons: clusterPersons,
      );
      clusters[clusterIds[i]] = newCluster;
    }

    update();
    return NotifierState.SUCCESS;
  }

  /// Cluster all the persons using K-means
  ///
  /// The [id] and [color] are assigned by the order.
  /// The [id] of each cluster is set to each [PersonModel] in the
  /// cluster
  Future<NotifierState> dbscanClustering(
      {double eps = 0.2, int min_samples = 10}) async {
    Map<String, dynamic> result = await repositoryDbscanClustering(
        selectedDatasetId, coords, eps, min_samples);

    final List<String> keys = result.keys.toList();
    final List<String> clusterIds = List.generate(
        keys.length, (index) => "cluster ${int.parse(keys[index])}");
    clusters = {};
    for (var i = 0; i < clusterIds.length; i++) {
      final List<String> clusterElements = List<String>.from(result[keys[i]]);
      final List<PersonModel> clusterPersons = [];
      for (var j = 0; j < clusterElements.length; j++) {
        final PersonModel personModel = personModelById(clusterElements[j]);
        personModel.clusterId = clusterIds[i];
        clusterPersons.add(personModel);
      }
      final ClusterModel newCluster = ClusterModel(
        id: clusterIds[i],
        color: uiUtilGetClusteringColor(i),
        persons: clusterPersons,
      );
      clusters[clusterIds[i]] = newCluster;
    }

    update();
    return NotifierState.SUCCESS;
  }

  /// Cluster all the persons by the categorical label selected
  ///
  /// The [id] is the name of the category, the color is given by order
  /// The [id] of each cluster is set to each [PersonModel] in the
  /// cluster
  Future<NotifierState> labeledClustering(String filter) async {
    final List<String> labels =
        List<String>.from(datasetSettings.categoricalData[filter]);
    clusters = {};
    for (var i = 0; i < labels.length; i++) {
      clusters[labels[i]] = ClusterModel(
        id: labels[i],
        color: uiUtilGetClusteringColor(i),
        persons: [],
      );
    }

    final int labelPosition = persons.first.categoricalLabels
        .indexWhere((element) => element == filter);

    for (var i = 0; i < persons.length; i++) {
      final String personLabel = persons[i].categoricalValues[labelPosition];
      clusters[personLabel].persons.add(persons[i]);
      persons[i].clusterId = personLabel;
    }

    update();
    return NotifierState.SUCCESS;
  }

  /// Adds one cluster from the given points
  ///
  /// the cluster [id] is given by the current clusters size
  /// the persons [clusterId] is set to this generated cluster [id]
  void addCustomCluster(List<InteractivePoint> selectedPoints) {
    final List<PersonModel> persons = [];
    if (selectedPoints.isNotEmpty) {
      final String clusterId = "cluster ${clusters.length}";

      for (int i = 0; i < selectedPoints.length; i++) {
        persons.add(selectedPoints[i].personModel);
        selectedPoints[i].personModel.clusterId = clusterId;
      }

      final ClusterModel newCluster = ClusterModel(
        id: clusterId,
        color: uiUtilGetClusteringColor(selectedPoints.length),
        persons: persons,
      );
      clusters[clusterId] = newCluster;
      update();
    }
  }

  /// Removes all clusters
  ///
  /// Set all [id]s of clusters to null of all [persons]
  void removeClusters() {
    clusters = {};
    variablesNamesOrdered = null;
    clusterIdA = null;
    clusterIdB = null;
    for (var i = 0; i < persons.length; i++) {
      persons[i].clusterId = null;
    }
    update();
  }

  Future<NotifierState> getOverview() async {
    Map<String, dynamic> data =
        await repositoryGetTemporalOverview(selectedDatasetId);
    overview = {};

    Map<String, List<double>> minOverview = {};
    Map<String, List<double>> maxOverview = {};
    Map<String, List<double>> meanOverview = {};
    for (var i = 0; i < datasetSettings.variablesNames.length; i++) {
      String varName = datasetSettings.variablesNames[i];
      minOverview[varName] = List<double>.from(data["min"][varName]);
      maxOverview[varName] = List<double>.from(data["max"][varName]);
      meanOverview[varName] = List<double>.from(data["mean"][varName]);
    }
    overview[OverviewType.MIN] = minOverview;
    overview[OverviewType.MAX] = maxOverview;
    overview[OverviewType.MEAN] = meanOverview;
    return NotifierState.SUCCESS;
  }

  void unselectDataset() {
    selectedDatasetId = null;
    redCluster = [];
    blueCluster = [];
    clusters = {};
    variablesNamesOrdered = null;
    datasetSettings = null;
    overview = null;
    d_k = null;
    coords = null;
    _persons = null;
    clusterIdA = null;
    clusterIdB = null;
  }
}
