import 'package:emotion_vis_client/controllers/series_controller.dart';
import 'package:emotion_vis_client/enums/app_enums.dart';
import 'package:emotion_vis_client/routes/route_names.dart';
import 'package:get/get.dart';

class SplashScreenUiController extends GetxController {
  @override
  void onReady() {
    startSettings();
    super.onReady();
  }

  Future<void> startSettings() async {
    SeriesController seriesController = Get.find();
    print("querying");
    NotifierState state = await seriesController.getDatasetsInfo();
    print("done");
    if (state != NotifierState.SUCCESS) {
      print("Can't get datasets information");
    }
    // try {
    //   await seriesController.getDatasetsInfo();
    // } catch (e) {
    //   uiUtilDialog(
    //     Container(
    //       height: 120,
    //       width: 120,
    //       alignment: Alignment.center,
    //       child: Column(
    //         children: [
    //           SizedBox(height: 30),
    //           Text(
    //             "Can't get datasets information from server",
    //             style: TextStyle(
    //               color: pTextColorSecondary,
    //               fontSize: 16,
    //             ),
    //             textAlign: TextAlign.center,
    //           ),
    //           POutlinedButton(
    //             text: "retry",
    //             onPressed: () {
    //               Get.back();
    //               startSettings();
    //             },
    //           ),
    //         ],
    //       ),
    //     ),
    //     dismissible: false,
    //   );
    //   return null;
    // }
    // print("route");
    route();
  }

  void route() {
    Get.offNamed(routePrincipalMenu);
  }
}
