import 'package:emotion_vis_client/interfaces/modules/home/home_ui.dart';
import 'package:emotion_vis_client/interfaces/modules/preprocessing/preprocessing_ui.dart';
import 'package:emotion_vis_client/interfaces/modules/principal_menu/principal_menu_ui.dart';
import 'package:emotion_vis_client/interfaces/modules/settings/settings_ui.dart';
import 'package:emotion_vis_client/interfaces/modules/single_person/single_person_ui.dart';
import 'package:emotion_vis_client/interfaces/modules/splash_screen/splash_screen_ui.dart';
import 'package:emotion_vis_client/interfaces/ui_utils.dart';
import 'package:emotion_vis_client/models/person_model.dart';
import 'package:emotion_vis_client/routes/route_bindings.dart';
import 'package:emotion_vis_client/routes/route_names.dart';
import 'package:get/get.dart';

List<GetPage> routePages = [
  GetPage(
    name: routeSplashScreen,
    page: () => const SplashScreen(),
    binding: SplashScreenBinding(),
  ),
  GetPage(
    name: routeHome,
    page: () => const HomeUi(),
    binding: HomeBinding(),
  ),
  GetPage(
    name: routePrincipalMenu,
    page: () => const PrincipalMenuUi(),
    binding: PrincipalMenuBinding(),
  ),
  GetPage(
    name: routeSettings,
    page: () => const SettingsUi(),
    binding: SettingsBinding(),
  ),
  GetPage(
    name: routePreprocessing,
    page: () => const PreprocessingUi(),
    binding: PreprocessingBinding(),
  ),
  GetPage(
    name: routeSinglePerson,
    page: () => SinglePersonUi(
      person: uiUtilGetArgument<PersonModel>(0),
    ),
    binding: SinglePersonBinding(),
  ),
];
