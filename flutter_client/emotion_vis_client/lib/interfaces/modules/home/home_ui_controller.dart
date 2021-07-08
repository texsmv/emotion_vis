import 'package:emotion_vis_client/app_constants.dart';
import 'package:emotion_vis_client/controllers/series_controller.dart';
import 'package:emotion_vis_client/interfaces/constants/colors.dart';
import 'package:emotion_vis_client/interfaces/general_widgets/dialogs/pdialog_proceed.dart';
import 'package:emotion_vis_client/interfaces/ui_utils.dart';
import 'package:emotion_vis_client/models/dataset_settings.dart';
import 'package:emotion_vis_client/models/visualization_levels.dart';
import 'package:get/get.dart';

class HomeUiController extends GetxController {
  final SeriesController _seriesController = Get.find();
  int _stackIndex = 0;
  set stackIndex(int value) {
    _stackIndex = value;
    update();
  }

  int get stackIndex => _stackIndex;

  Map<String, List<double>> get overview => _seriesController
      .overview[_seriesController.datasetSettings.overviewType];
  DatasetSettings get datasetSettings => _seriesController.datasetSettings;
  MultipleView selectedVisualization = MultipleView.values[0];
  List<TemporalVisualization> availableTemporalVisualizations;
  @override
  void onInit() {
    super.onInit();
    availableTemporalVisualizations = uiUtilAvailableTemporalVisualizations(
      datasetSettings.isTagged,
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

  TemporalVisualization temporalVisualization;
  NonTemporalVisualization nonTemporalVisualization;

  void changeDistance(int value) {
    _seriesController.updateSettings(distance: value);
  }

  void changeProjection(int value) {
    _seriesController.updateSettings(projection: value);
  }

  void changeProjectionParameter(int value) {
    _seriesController.updateSettings(projectionParameter: value);
  }

  void updateRange() {
    _seriesController
        .updateSettings(
      windowPosition: datasetSettings.windowPosition,
      windowLength: datasetSettings.windowLength,
    )
        .then(
      (value) {
        if (_seriesController.clusterIdA != null &&
            _seriesController.clusterIdB != null) {
          _seriesController.getFishersDiscriminantRanking(
              _seriesController.clusterIdA, _seriesController.clusterIdB);
        }
      },
    );
  }

  Future<void> changeAllView(MultipleView view) async {
    if (view == MultipleView.GRID) {
      if (datasetSettings.labels.length > MAX_TIME_POINTS_GRIDVIEW) {
        await uiUtilDialog(
          PDialogProceed(
            color: pColorPrimary,
            onProceed: () {
              selectedVisualization = view;
              Get.back();
            },
            onCancel: () {
              Get.back();
            },
            title: "Alert",
            description:
                "View not recommended for time series with a size higher than $MAX_TIME_POINTS_GRIDVIEW. You can try using a window size lower than $MAX_TIME_POINTS_GRIDVIEW.",
            proceedButtonText: "continue",
            cancelButtonText: "cancel",
          ),
        );
      } else {
        selectedVisualization = view;
      }
    } else {
      selectedVisualization = view;
    }
    update();
  }
}
