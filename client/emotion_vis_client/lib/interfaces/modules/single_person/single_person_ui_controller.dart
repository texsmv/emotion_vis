import 'package:emotion_vis_client/controllers/series_controller.dart';
import 'package:emotion_vis_client/interfaces/ui_utils.dart';
import 'package:emotion_vis_client/interfaces/visualizations/vis_settings.dart';
import 'package:emotion_vis_client/models/dataset_settings.dart';
import 'package:emotion_vis_client/models/person_model.dart';
import 'package:emotion_vis_client/models/visualization_levels.dart';
import 'package:get/get.dart';

class SinglePersonUiController extends GetxController {
  final SeriesController _seriesController = Get.find();
  DatasetSettings get datasetSettings => _seriesController.datasetSettings;
  String personId;
  PersonModel personModel;
  bool _firstBuild = true;
  TemporalVisualization temporalVisualization;
  NonTemporalVisualization nonTemporalVisualization;
  List<TemporalVisualization> availableTemporalVisualizations;
  List<NonTemporalVisualization> availableNonTemporalVisualizations;
  int timePoint = 0;

  void initState(PersonModel person) {
    if (!_firstBuild) return;
    _firstBuild = false;
    this.personModel = person;
    this.personId = person.id;

    availableTemporalVisualizations = uiUtilAvailableTemporalVisualizations(
      datasetSettings.isTagged,
      datasetSettings.modelType,
      datasetSettings.variablesLength,
    );
    availableNonTemporalVisualizations =
        uiUtilAvailableNonTemporalVisualizations(
      datasetSettings.modelType,
      datasetSettings.variablesLength,
    );

    temporalVisualization = uiUtilAvailableTemporalVisualizations(
        datasetSettings.isTagged,
        datasetSettings.modelType,
        datasetSettings.variablesLength)[0];
    nonTemporalVisualization = uiUtilAvailableNonTemporalVisualizations(
        datasetSettings.modelType, datasetSettings.variablesLength)[0];
  }
}
