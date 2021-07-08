import 'dart:ui';

import 'package:emotion_vis_client/app_constants.dart';
import 'package:emotion_vis_client/controllers/series_controller.dart';
import 'package:emotion_vis_client/interfaces/constants/colors.dart';
import 'package:emotion_vis_client/interfaces/general_widgets/buttons/pfilled_button.dart';
import 'package:emotion_vis_client/interfaces/general_widgets/buttons/poutlined_button.dart';
import 'package:emotion_vis_client/interfaces/general_widgets/containers/pcard.dart';
import 'package:emotion_vis_client/interfaces/modules/principal_menu/components/welcome_side.dart';
import 'package:emotion_vis_client/interfaces/modules/settings/components/time_serie_tile.dart';
import 'package:emotion_vis_client/interfaces/modules/settings/settings_ui_controller.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:glass_kit/glass_kit.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:prefab_animations/event_animation/animations/bounce_animation.dart';
import 'package:prefab_animations/event_animation/event_animation.dart';

class SettingsUi extends GetView<SettingsUiController> {
  const SettingsUi({Key key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        width: double.infinity,
        height: double.infinity,
        decoration: const BoxDecoration(
          image: DecorationImage(
            image: AssetImage(ASSET_WALLPAPER),
            fit: BoxFit.cover,
          ),
        ),
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 10.0, sigmaY: 10.0),
          child: GlassContainer.clearGlass(
            borderColor: Colors.transparent,
            height: 600,
            width: 1250,
            child: Row(
              children: [
                const WelcomeSide(),
                Expanded(
                  child: GetBuilder<SettingsUiController>(
                    builder: (_) => SizedBox(
                      width: 700,
                      height: 600,
                      child: PCard(
                        child: Container(
                          padding: EdgeInsets.all(15),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Align(
                                alignment: Alignment.center,
                                child: const Text(
                                  "Dataset settings",
                                  style: TextStyle(
                                    fontSize: 26,
                                    fontWeight: FontWeight.w600,
                                    color: pColorAccent,
                                  ),
                                ),
                              ),
                              OptionWindowSize(),
                              const Divider(
                                color: pColorAccent,
                                thickness: 1,
                              ),
                              DistanceOption(),
                              const Divider(
                                color: pColorAccent,
                                thickness: 1,
                              ),
                              ProjectionOption(),
                              const Divider(
                                color: pColorAccent,
                                thickness: 1,
                              ),
                              Expanded(
                                child: GetBuilder<SettingsUiController>(
                                    builder: (_) => OptionVariables()),
                              ),
                              Padding(
                                padding:
                                    const EdgeInsets.symmetric(horizontal: 5),
                                child: Row(
                                  mainAxisAlignment:
                                      MainAxisAlignment.spaceBetween,
                                  children: [
                                    SizedBox(
                                      width: 80,
                                      child: PFilledButton(
                                        buttonColor: pColorAccent,
                                        text: "Back",
                                        onPressed: () {
                                          Get.back();
                                        },
                                      ),
                                    ),
                                    EventAnimation(
                                      awaitAnimationDuration:
                                          const Duration(milliseconds: 600),
                                      awaitAnimationBuilder:
                                          (controller, child) {
                                        return BounceAnimation(
                                          minScale: 0.95,
                                          controller: controller,
                                          child: child,
                                        );
                                      },
                                      child: SizedBox(
                                        width: 110,
                                        child: PFilledButton(
                                          buttonColor: pColorDark,
                                          text: "Open",
                                          onPressed: controller.onApplySettings,
                                        ),
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _appLogo() {
    return Padding(
      padding: EdgeInsets.symmetric(vertical: 30),
      child: Text(
        "EmotionVis",
        style: GoogleFonts.lobster(
          fontSize: 48,
          color: pTextColorPrimary,
        ),
      ),
    );
  }
}

class OptionWindowSize extends GetView<SettingsUiController> {
  const OptionWindowSize({Key key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            const Text(
              "Explore using all time points:",
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w400,
              ),
            ),
            GetBuilder(
              builder: (_) => Switch(
                value: controller.visualizeAllTime,
                onChanged: (value) {
                  controller.visualizeAllTime = value;
                },
              ),
            )
          ],
        ),
        Visibility(
          visible: !controller.visualizeAllTime,
          child: Padding(
            padding: EdgeInsets.only(left: 20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text("Use a window size of:"),
                    Container(
                      width: 30,
                      child: TextField(
                        textAlign: TextAlign.center,
                        controller: controller.windowLengthController,
                      ),
                    ),
                  ],
                ),
                Text(
                  "* Number of time point that will be shown in the main view",
                  style: TextStyle(
                    color: pTextColorSecondary,
                    fontSize: 12,
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}

class DistanceOption extends GetView<SettingsUiController> {
  DistanceOption({Key key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        const Text(
          "Distance:",
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w400,
          ),
        ),
        Row(
          children: [
            PFilledButton(
              buttonColor: controller.selectedDistance == 0
                  ? pColorAccent
                  : Color.fromARGB(255, 240, 240, 240),
              textColor:
                  controller.selectedDistance == 0 ? Colors.white : pColorGray,
              text: "Euclidean",
              onPressed: () {
                controller.selectedDistance = 0;
                controller.update();
              },
            ),
            SizedBox(width: 10),
            PFilledButton(
              buttonColor: controller.selectedDistance == 1
                  ? pColorAccent
                  : Color.fromARGB(255, 240, 240, 240),
              textColor:
                  controller.selectedDistance == 1 ? Colors.white : pColorGray,
              text: "Dynamic time warping",
              onPressed: () {
                controller.selectedDistance = 1;
                controller.update();
              },
            ),
          ],
        ),
      ],
    );
  }
}

class ProjectionOption extends GetView<SettingsUiController> {
  ProjectionOption({Key key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        const Text(
          "Dimensionality reduction algoritm:",
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w400,
          ),
        ),
        Row(
          children: [
            PFilledButton(
              buttonColor: controller.selectedProjection == 0
                  ? pColorAccent
                  : Color.fromARGB(255, 240, 240, 240),
              textColor: controller.selectedProjection == 0
                  ? Colors.white
                  : pColorGray,
              text: "MDS",
              onPressed: () {
                controller.selectedProjection = 0;
                controller.update();
              },
            ),
            SizedBox(width: 10),
            PFilledButton(
              buttonColor: controller.selectedProjection == 1
                  ? pColorAccent
                  : Color.fromARGB(255, 240, 240, 240),
              textColor: controller.selectedProjection == 1
                  ? Colors.white
                  : pColorGray,
              text: "Isomap",
              onPressed: () {
                controller.selectedProjection = 1;
                controller.update();
              },
            ),
            SizedBox(width: 10),
            PFilledButton(
              buttonColor: controller.selectedProjection == 2
                  ? pColorAccent
                  : Color.fromARGB(255, 240, 240, 240),
              textColor: controller.selectedProjection == 2
                  ? Colors.white
                  : pColorGray,
              text: "UMAP",
              onPressed: () {
                controller.selectedProjection = 2;
                controller.update();
              },
            ),
            SizedBox(width: 10),
            PFilledButton(
              buttonColor: controller.selectedProjection == 3
                  ? pColorAccent
                  : Color.fromARGB(255, 240, 240, 240),
              textColor: controller.selectedProjection == 3
                  ? Colors.white
                  : pColorGray,
              text: "T-SNE",
              onPressed: () {
                controller.selectedProjection = 3;
                controller.update();
              },
            ),
          ],
        ),
      ],
    );
  }
}

class OptionVariables extends GetView<SettingsUiController> {
  OptionVariables({Key key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const SizedBox(height: 10),
        const Text(
          "Emotion dimensions colors:",
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w400,
          ),
        ),
        Expanded(
          child: SizedBox(
            width: 500,
            child: ListView.separated(
              separatorBuilder: (context, index) => const Divider(
                height: 2,
              ),
              shrinkWrap: true,
              padding: const EdgeInsets.only(left: 30),
              itemBuilder: (context, index) {
                return TimeSerieTile(
                    varName: controller.datasetSettings.variablesNames[index]);
              },
              itemCount: controller.datasetSettings.variablesLength,
            ),
          ),
        ),
      ],
    );
  }
}
