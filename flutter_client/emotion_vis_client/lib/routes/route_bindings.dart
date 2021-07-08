import 'package:emotion_vis_client/controllers/series_controller.dart';
import 'package:emotion_vis_client/interfaces/modules/home/home_ui_controller.dart';
import 'package:emotion_vis_client/interfaces/modules/preprocessing/preprocessing_ui_controller.dart';
import 'package:emotion_vis_client/interfaces/modules/principal_menu/principal_menu_ui_controller.dart';
import 'package:emotion_vis_client/interfaces/modules/projection_view/projection_view_ui_controller.dart';
import 'package:emotion_vis_client/interfaces/modules/settings/settings_ui_controller.dart';
import 'package:emotion_vis_client/interfaces/modules/single_person/single_person_ui_controller.dart';
import 'package:emotion_vis_client/interfaces/modules/splash_screen/splash_screen_ui_controller.dart';
import 'package:emotion_vis_client/interfaces/modules/visualizations_view/visualization_view_ui_controller.dart';
import 'package:get/get.dart';

class SplashScreenBinding implements Bindings {
  @override
  void dependencies() {
    Get.put(SeriesController(), permanent: true);
    // Get.put(SeriesController(), permanent: true);
    Get.put(SplashScreenUiController());
  }
}

class HomeBinding implements Bindings {
  @override
  void dependencies() {
    Get.lazyPut(() => HomeUiController());
    Get.lazyPut(() => ProjectionViewUiController());
    Get.lazyPut(() => VisualizationsViewUiController());
  }
}

class SettingsBinding implements Bindings {
  @override
  void dependencies() {
    Get.delete<SettingsUiController>();
    Get.lazyPut(() => SettingsUiController());
  }
}

class PreprocessingBinding implements Bindings {
  @override
  void dependencies() {
    Get.lazyPut(() => PreprocessingUiController());
  }
}

class PrincipalMenuBinding implements Bindings {
  @override
  void dependencies() {
    Get.lazyPut(() => PrincipalMenuUiController());
  }
}

class SinglePersonBinding implements Bindings {
  @override
  void dependencies() {
    Get.lazyPut(() => SinglePersonUiController());
  }
}
