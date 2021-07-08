import 'package:emotion_vis_client/enums/app_enums.dart';
import 'package:emotion_vis_client/interfaces/constants/colors.dart';
import 'package:emotion_vis_client/interfaces/general_widgets/containers/pcard.dart';
import 'package:emotion_vis_client/interfaces/modules/preprocessing/preprocessing_ui_controller.dart';
import 'package:emotion_vis_client/utils.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:menu_button/menu_button.dart';

class PreprocessingUi extends GetView<PreprocessingUiController> {
  const PreprocessingUi({Key key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(),
      body: Container(
        width: double.infinity,
        height: double.infinity,
        child: SingleChildScrollView(
          child: Column(
            children: [
              Padding(
                padding: EdgeInsets.symmetric(vertical: 30),
                child: Text(
                  "EmotionVis",
                  style:
                      GoogleFonts.lobster(fontSize: 48, color: pColorPrimary),
                ),
              ),
              SizedBox(
                width: 700,
                height: 600,
                child: PCard(
                    child: Container(
                  padding: EdgeInsets.all(20),
                  child: Column(
                    children: [
                      Divider(
                        color: pColorAccent,
                      ),
                      // downsampling
                      Visibility(
                        visible: controller.showDateOptions,
                        child: Container(
                          height: 80,
                          child: Column(
                            children: [
                              Row(
                                mainAxisAlignment:
                                    MainAxisAlignment.spaceBetween,
                                children: [
                                  Text("Allowed downsample rules:"),
                                  Obx(
                                    () => MenuButton<DownsampleRule>(
                                      child: Container(
                                        width: 120,
                                        height: 30,
                                        color: pColorPrimary,
                                        alignment: Alignment.center,
                                        padding: const EdgeInsets.symmetric(
                                            horizontal: 16),
                                        child: Text(
                                          Utils.downsampleRule2Str(
                                              controller.selectedRule.value),
                                          textAlign: TextAlign.center,
                                          style: TextStyle(
                                              color: Colors.white,
                                              fontWeight: FontWeight.w400),
                                        ),
                                      ),
                                      items: controller.allowedDownsampleRules,
                                      topDivider: true,
                                      scrollPhysics:
                                          AlwaysScrollableScrollPhysics(),
                                      onItemSelected: (DownsampleRule value) {
                                        controller.selectedRule.value = value;
                                      },
                                      onMenuButtonToggle: (isToggle) {},
                                      decoration: BoxDecoration(
                                        border: Border.all(
                                            color: Colors.white.withAlpha(0)),
                                        borderRadius: const BorderRadius.all(
                                            Radius.circular(3.0)),
                                        color: Colors.white.withAlpha(0),
                                      ),
                                      divider: Container(
                                        height: 1,
                                        color: Colors.grey,
                                      ),
                                      toggledChild: Container(
                                        height: 30,
                                      ),
                                      itemBuilder: (DownsampleRule value) =>
                                          Container(
                                              width: 80,
                                              height: 30,
                                              color: Colors.white,
                                              alignment: Alignment.centerLeft,
                                              child: Text(
                                                  Utils.downsampleRule2Str(
                                                      value))),
                                    ),
                                  ),
                                ],
                              ),
                            ],
                          ),
                        ),
                      ),
                      Divider(
                        color: pColorAccent,
                      ),
                    ],
                  ),
                )),
              ),
            ],
          ),
        ),
      ),
      floatingActionButton: FloatingActionButton.extended(
        backgroundColor: Get.theme.primaryColor,
        heroTag: "float",
        onPressed: controller.onApplyChanges,
        label: Text(
          "Next",
          style: TextStyle(color: Colors.white),
        ),
      ),
    );
  }
}
