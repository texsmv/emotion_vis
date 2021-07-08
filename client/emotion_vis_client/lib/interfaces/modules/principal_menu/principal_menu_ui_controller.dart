import 'package:emotion_vis_client/controllers/series_controller.dart';
import 'package:emotion_vis_client/interfaces/constants/colors.dart';
import 'package:emotion_vis_client/interfaces/ui_utils.dart';
import 'package:emotion_vis_client/models/dataset_settings.dart';
import 'package:emotion_vis_client/repositories/series_repository.dart';
import 'package:emotion_vis_client/routes/route_names.dart';
import 'package:emotion_vis_client/utils.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:emotion_vis_client/enums/app_enums.dart';

class PrincipalMenuUiController extends GetxController {
  final SeriesController _seriesController = Get.find();
  TextEditingController datasetNameController = TextEditingController();
  ScrollController scrollController = ScrollController();

  List<String> get loadedDatasetsIds => _seriesController.loadedDatasetsIds;
  List<String> get localDatasetsIds => _seriesController.localDatasetsIds;
  int get nLoadedDatasets => loadedDatasetsIds.length;

  Future<void> selectDataset(String datasetId) async {
    _seriesController.selectedDatasetId = datasetId;
    await _seriesController.getDatasetInfo();

    Get.toNamed(routeSettings);
  }

  Map<String, DatasetSettings> datasetsSettings;
  @override
  void onInit() {
    loadDatasetsInfo();
    super.onInit();
  }

  Future<void> loadDatasetsInfo() async {
    final Map<String, DatasetSettings> datasets = {};
    for (var i = 0; i < loadedDatasetsIds.length; i++) {
      Map<String, dynamic> infoMap =
          await repositoryGetDatasetInfo(loadedDatasetsIds[i]);
      DatasetSettings settings = DatasetSettings.fromMap(info: infoMap);

      datasets[loadedDatasetsIds[i]] = settings;
    }
    datasetsSettings = datasets;
    update();
  }

  // TODO enable later
  Future<void> addDataset() async {
    await FilePicker.platform.pickFiles(allowMultiple: false);
    uiUtilShowLoaderOverlay();
    final String configData = await pickConfigurationFile();
    print("1");
    if (configData == null) {
      uiUtilShowMessage("Couldn't open the configuration file.");
    }
    print("2");
    print(configData != null);
    await Future.delayed(Duration(milliseconds: 1200));
    List<String> emotionFilesData =
        configData != null ? await pickJsonFiles() : null;
    await Future.delayed(Duration(milliseconds: 1200));
    print("3");
    if (emotionFilesData == null) {
      uiUtilShowMessage("No emotion file opened, we couldn't add the dataset.");
    }
    print("4");

    if (configData != null && emotionFilesData != null) {
      final String datasetId =
          await _seriesController.initializeDataset(configData);
      for (var i = 0; i < emotionFilesData.length; i++) {
        await _seriesController.addEml(datasetId, emotionFilesData[i]);
      }

      await _seriesController.getDatasetsInfo();
      await loadDatasetsInfo();
    }
    print("5");

    uiUtilHideLoaderOverlay();
  }

  Future<void> removeDataset(String datasetId) async {
    await _seriesController.removeDataset(datasetId);
    await _seriesController.getDatasetsInfo();
    await loadDatasetsInfo();
    update();
  }

  Future<void> preprocessDataset(String datasetId) async {
    _seriesController.selectedDatasetId = datasetId;
    await _seriesController.getDatasetInfo();
    if (!_seriesController.datasetSettings.isDated) {
      Get.snackbar(
          "Downsampling", "you need to provide datetimes for your data");
      return;
    }
    List<DownsampleRule> allowedDownsampleRules = List.generate(
        _seriesController.datasetSettings.downsampleRules.length,
        (index) => Utils.str2downsampleRule(
            _seriesController.datasetSettings.downsampleRules[index]));
    print(allowedDownsampleRules);
    uiUtilDialog(
      Container(
        height: 300,
        width: 300,
        child: Column(
          children: [
            SizedBox(height: 20),
            Text(
              "Downsample options",
              style: TextStyle(
                fontSize: 16,
                color: pTextColorSecondary,
                fontWeight: FontWeight.w600,
              ),
            ),
            SizedBox(height: 10),
            Expanded(
              child: ListView.separated(
                separatorBuilder: (context, index) => Divider(height: 2),
                itemCount: allowedDownsampleRules.length,
                itemBuilder: (context, index) {
                  return GestureDetector(
                    behavior: HitTestBehavior.opaque,
                    onTap: () async {
                      uiUtilShowLoaderOverlay();
                      await _seriesController
                          .downsampleData(allowedDownsampleRules[index]);
                      await loadDatasetsInfo();
                      await _seriesController.getDatasetInfo();
                      uiUtilHideLoaderOverlay();
                      Get.back();
                    },
                    child: Container(
                      height: 50,
                      alignment: Alignment.center,
                      child: Text(
                        Utils.downsampleRule2UiStr(
                            allowedDownsampleRules[index]),
                      ),
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
    // Get.toNamed(routePreprocessing);
  }

  Future<String> pickDatasetInfo() async {
    final FilePickerResult result =
        await FilePicker.platform.pickFiles(allowMultiple: false);
    // final List<File> files = result.paths.map((path) => File(path)).toList();
    if (result == null) return null;
    if (result.files.isNotEmpty) {
      String datasetInfoJson = String.fromCharCodes(result.files.first.bytes);
      return _seriesController.initializeDataset(datasetInfoJson);
    }
    return null;
  }

  /// Picks the dataset configuration file
  ///
  /// Returns [null] if the file was not succesfully picked
  Future<String> pickConfigurationFile() async {
    final FilePickerResult result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ["json"],
    );
    if (result == null) return null;
    if (result.files.isNotEmpty) {
      return String.fromCharCodes(result.files.first.bytes);
    }
    return null;
  }

  Future<void> pickEmotionFiles(String datasetId) async {
    final FilePickerResult result =
        await FilePicker.platform.pickFiles(allowMultiple: true);
    if (result == null) return null;
    if (result.files.isNotEmpty) {
      for (int i = 0; i < result.files.length; i++) {
        String xmlString = String.fromCharCodes(result.files[i].bytes);
        await _seriesController.addEml(datasetId, xmlString);
      }
    }
  }

  Future<List<String>> pickJsonFiles() async {
    List<String> jsonStrings = [];
    FilePickerResult result = await FilePicker.platform.pickFiles(
      allowMultiple: true,
      type: FileType.custom,
      allowedExtensions: ["json"],
    );

    print(result);
    if (result == null) return null;
    if (result.files.isNotEmpty) {
      for (int i = 0; i < result.files.length; i++) {
        String data = String.fromCharCodes(result.files[i].bytes);
        jsonStrings.add(data);
      }
      return jsonStrings;
    }
    return null;
  }

  void openLocalDataset() {
    uiUtilDialog(
      Container(
        height: 400,
        width: 300,
        padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 20),
        child: Column(
          children: [
            const Text(
              "Open:",
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w600,
                color: pColorPrimary,
              ),
            ),
            Expanded(
              child: ListView.separated(
                separatorBuilder: (context, index) => const Divider(height: 10),
                itemCount: localDatasetsIds.length,
                itemBuilder: (context, index) {
                  return GestureDetector(
                    behavior: HitTestBehavior.opaque,
                    onTap: () async {
                      uiUtilShowLoaderOverlay();
                      NotifierState state = await _seriesController
                          .loadLocalDataset(localDatasetsIds[index]);
                      if (state == NotifierState.SUCCESS) {
                        await _seriesController.getDatasetsInfo();
                        Get.back();
                      }
                      print(state);
                      await loadDatasetsInfo();
                      uiUtilHideLoaderOverlay();
                    },
                    child: Container(
                      height: 40,
                      alignment: Alignment.center,
                      child: Text(localDatasetsIds[index]),
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}
