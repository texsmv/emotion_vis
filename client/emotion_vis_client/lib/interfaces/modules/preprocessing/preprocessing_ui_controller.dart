import 'package:emotion_vis_client/controllers/series_controller.dart';
import 'package:emotion_vis_client/enums/app_enums.dart';
import 'package:emotion_vis_client/models/dataset_settings.dart';
import 'package:emotion_vis_client/routes/route_names.dart';
import 'package:emotion_vis_client/utils.dart';
import 'package:get/state_manager.dart';
import 'package:get/get.dart';

class PreprocessingUiController extends GetxController {
  final SeriesController _seriesController = Get.find();
  Rx<DownsampleRule> selectedRule = DownsampleRule.NONE.obs;

  bool get showDateOptions => _seriesController.datasetSettings.isDated;
  DatasetSettings get datasetSettings => _seriesController.datasetSettings;
  List<DownsampleRule> get allowedDownsampleRules => List.generate(
      _seriesController.datasetSettings.downsampleRules.length,
      (index) => Utils.str2downsampleRule(
          _seriesController.datasetSettings.downsampleRules[index]));

  Future<void> onApplyChanges() async {
    if (datasetSettings.isDated && selectedRule.value != DownsampleRule.NONE) {
      await _seriesController.downsampleData(selectedRule.value);
      await _seriesController.getDatasetInfo();
      Get.toNamed(routeSettings);
    }
  }
}
