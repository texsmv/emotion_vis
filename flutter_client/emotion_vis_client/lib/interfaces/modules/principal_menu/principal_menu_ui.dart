import 'dart:ui';

import 'package:emotion_vis_client/app_constants.dart';
import 'package:emotion_vis_client/interfaces/constants/colors.dart';
import 'package:emotion_vis_client/interfaces/modules/principal_menu/principal_menu_ui_controller.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:glass_kit/glass_kit.dart';

import 'components/datasets_selection_side.dart';
import 'components/welcome_side.dart';

class PrincipalMenuUi extends GetView<PrincipalMenuUiController> {
  const PrincipalMenuUi({Key key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: pColorScaffold,
      body: Container(
        width: double.infinity,
        height: double.infinity,
        decoration: const BoxDecoration(
          image: DecorationImage(
            image: AssetImage(ASSET_WALLPAPER),
            fit: BoxFit.cover,
          ),
        ),
        alignment: Alignment.center,
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 10.0, sigmaY: 10.0),
          child: GlassContainer.clearGlass(
            borderColor: Colors.transparent,
            height: 600,
            width: 1250,
            elevation: 3,
            child: Row(
              children: const [
                WelcomeSide(),
                Expanded(
                  child: DatasetSelectionSide(),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
