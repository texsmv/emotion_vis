import 'dart:core';
import 'dart:math';

import 'package:emotion_vis_client/controllers/series_controller.dart';
import 'package:emotion_vis_client/list_extension.dart';
import 'package:emotion_vis_client/enums/app_enums.dart';
import 'package:emotion_vis_client/models/visualization_levels.dart';
import 'package:flutter/material.dart';
import 'package:flutter_colorpicker/flutter_colorpicker.dart';
import 'package:get/get.dart';

class Utils {
  static String downsampleRule2Str(DownsampleRule rule) {
    switch (rule) {
      case DownsampleRule.YEARS:
        return "A";
      case DownsampleRule.MONTHS:
        return "M";
      case DownsampleRule.DAYS:
        return "D";
      case DownsampleRule.HOURS:
        return "H";
      case DownsampleRule.MINUTES:
        return "T";
      case DownsampleRule.SECONDS:
        return "S";
      case DownsampleRule.NONE:
        return "NONE";
      default:
        return "NONE";
    }
  }

  static String downsampleRule2UiStr(DownsampleRule rule) {
    switch (rule) {
      case DownsampleRule.YEARS:
        return "years";
      case DownsampleRule.MONTHS:
        return "months";
      case DownsampleRule.DAYS:
        return "days";
      case DownsampleRule.HOURS:
        return "hours";
      case DownsampleRule.MINUTES:
        return "minutes";
      case DownsampleRule.SECONDS:
        return "seconds";
      case DownsampleRule.NONE:
        return "NONE";
      default:
        return "NONE";
    }
  }

  static DownsampleRule str2downsampleRule(String rule) {
    switch (rule) {
      case "A":
        return DownsampleRule.YEARS;
      case "M":
        return DownsampleRule.MONTHS;
      case "D":
        return DownsampleRule.DAYS;
      case "H":
        return DownsampleRule.HOURS;
      case "T":
        return DownsampleRule.MINUTES;
      case "S":
        return DownsampleRule.SECONDS;
        break;
      default:
        return DownsampleRule.SECONDS;
    }
  }
}

Offset polarToCartesian(double angle, double r) {
  return Offset(r * cos(angle), r * sin(angle));
}

String dateTimeHour2Str(DateTime date) {
  String minutes = timeDigit2Str(date.minute);
  String seconds = timeDigit2Str(date.second);

  return minutes + ":" + seconds;
}

String timeDigit2Str(int value) {
  if (value >= 10)
    return value.toString();
  else
    return "0" + value.toString();
}

Future<Color> pickColor(Color pickerColor) async {
  Color pickedColor = Colors.black;
  await showDialog(
    context: Get.context,
    builder: (_) => AlertDialog(
      title: const Text('Pick a color!'),
      content: SingleChildScrollView(
        child: ColorPicker(
          pickerColor: pickerColor,
          onColorChanged: (Color newColor) {
            pickedColor = newColor;
          },
          showLabel: true,
          pickerAreaHeightPercent: 0.8,
        ),
      ),
      actions: <Widget>[
        FlatButton(
          child: const Text('Got it'),
          onPressed: () {
            Get.back();
          },
        ),
      ],
    ),
  );
  return pickedColor;
}

bool isNumeric(String s) {
  if (s == null) {
    return false;
  }
  return double.tryParse(s) != null;
}

double rangeConverter(double oldValue, double oldMin, double oldMax,
    double newMin, double newMax) {
  return (((oldValue - oldMin) * (newMax - newMin)) / (oldMax - oldMin)) +
      newMin;
}
