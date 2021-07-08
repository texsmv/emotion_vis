import 'package:emotion_vis_client/interfaces/constants/colors.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class WelcomeSide extends StatelessWidget {
  const WelcomeSide({Key key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 300,
      height: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: Column(
        children: [
          const SizedBox(height: 40),
          Text(
            "EmotionVis",
            style: GoogleFonts.lobster(
              fontSize: 48,
              // color: pTextColorWhite,
              color: pColorAccent,
            ),
          ),
          const SizedBox(height: 20),
          const Text(
            "Welcome to EmotionVis, a web base tool to visualize emotion's temporal data.",
            style: TextStyle(
              fontSize: 18,
              color: pColorDark,
            ),
          ),
        ],
      ),
    );
  }
}
