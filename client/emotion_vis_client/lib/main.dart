import 'package:emotion_vis_client/interfaces/constants/colors.dart';
import 'package:emotion_vis_client/interfaces/modules/splash_screen/splash_screen_ui.dart';
import 'package:emotion_vis_client/routes/route_names.dart';
import 'package:emotion_vis_client/routes/router.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:get/get_navigation/src/root/get_material_app.dart';
import 'package:loader_overlay/loader_overlay.dart';

void main() {
  runApp(
    GlobalLoaderOverlay(
      overlayOpacity: 0.3,
      overlayColor: pColorPrimary,
      useDefaultLoading: true,
      child: GetMaterialApp(
        defaultTransition: Transition.fade,
        theme: ThemeData(
          scaffoldBackgroundColor: pColorScaffold,
          primaryColor: pColorPrimary,
          accentColor: pColorAccent,
          // textTheme: GoogleFonts.ralewayTextTheme(
          //   Get.textTheme,
          // ),
        ),
        onInit: loadInitialControllers(),
        getPages: routePages,
        initialRoute: routeSplashScreen,
        onUnknownRoute: (settings) {
          return MaterialPageRoute(
            builder: (context) {
              return SplashScreen();
            },
          );
        },
        // initialRoute: routePrincipalMenu,
      ),
    ),
  );
}

dynamic loadInitialControllers() {
  // Get.put(SeriesController(), permanent: true);
}
