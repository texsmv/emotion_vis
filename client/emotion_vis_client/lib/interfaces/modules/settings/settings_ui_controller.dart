import 'package:emotion_vis_client/controllers/series_controller.dart';
import 'package:emotion_vis_client/interfaces/general_widgets/buttons/poutlined_button.dart';
import 'package:emotion_vis_client/interfaces/ui_utils.dart';
import 'package:emotion_vis_client/models/dataset_settings.dart';
import 'package:emotion_vis_client/routes/route_names.dart';
import 'package:flutter/material.dart';
import 'package:flutter_colorpicker/flutter_colorpicker.dart';
import 'package:get/get.dart';

class SettingsUiController extends GetxController {
  SeriesController _seriesController;
  TextEditingController windowLengthController = TextEditingController();
  int selectedDistance = 0;
  int selectedProjection = 0;
  int selectedParameter = 5;
  bool get visualizeAllTime => _seriesController.visualizeAllTime;
  set visualizeAllTime(bool value) {
    _seriesController.visualizeAllTime = value;
    if (true) {
      _seriesController.datasetSettings.windowLength =
          _seriesController.datasetSettings.timeLength;
    }
    update();
  }

  DatasetSettings get datasetSettings => _seriesController.datasetSettings;

  @override
  void onInit() {
    _seriesController = Get.find<SeriesController>();
    windowLengthController = TextEditingController(
        text: _seriesController.datasetSettings.windowLength.toString());
    selectedDistance = _seriesController.datasetSettings.distance;
    selectedProjection = _seriesController.datasetSettings.projection;
    selectedParameter = _seriesController.datasetSettings.projectionParameter;
    super.onInit();
  }

  void onApplySettings() async {
    int windowLength = double.parse(windowLengthController.text).toInt();
    if (windowLength <= 2) {
      Get.snackbar("Settings", "The window size must be greater than 2.");
      return;
    }
    if (windowLength > datasetSettings.timeLength) {
      Get.snackbar("Settings", "The window size can't exceed the total size.");
      return;
    }
    uiUtilShowLoaderOverlay();
    if (windowLength == datasetSettings.timeLength) {
      _seriesController.visualizeAllTime = true;
    }
    Get.snackbar("Opening dataset", "This may take some minutes.");
    await _seriesController.updateSettings(
      windowLength: double.parse(windowLengthController.text).toInt(),
      windowPosition: 0,
      colors: datasetSettings.variablesColors,
      distance: selectedDistance,
      projection: selectedProjection,
      projectionParameter: selectedParameter,
    );
    await _seriesController.getOverview();
    Get.offAllNamed(routeHome);
    update();
    uiUtilHideLoaderOverlay();
  }

  void editTimeSerieItem(String varName) async {
    Color newColor = datasetSettings.variablesColors[varName];
    await showDialog(
      context: Get.context,
      builder: (_) => AlertDialog(
        actions: [
          POutlinedButton(
            text: "Done",
            onPressed: () {
              Get.back();
            },
          )
        ],
        content: SizedBox(
          height: 500,
          child: ColorPicker(
            pickerColor: newColor,
            onColorChanged: (Color val) {
              newColor = val;
            },
            showLabel: true,
            pickerAreaHeightPercent: 0.8,
          ),
        ),
      ),
    );
    datasetSettings.variablesColors[varName] = newColor;
    update();
  }
}
