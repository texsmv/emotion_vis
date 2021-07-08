// import 'dart:math';

// import 'package:emotion_vis_client/interfaces/ui_utils.dart';
// import 'package:emotion_vis_client/interfaces/visualizations/vis_settings.dart';
// import 'package:emotion_vis_client/models/person_model.dart';
// import 'package:flutter/cupertino.dart';
// import 'package:flutter/material.dart';
// import 'package:get/get.dart';
// import 'package:touchable/touchable.dart';

// class MultipleGridViewPainter extends CustomPainter {
//   List<PersonModel> persons;
//   VisSettings visSettings;

//   MultipleGridViewPainter({
//     @required this.persons,
//     @required this.visSettings,
//   });

//   double _width;
//   double _height;
//   double _blockWidth;
//   Canvas _canvas;

//   int get timeLength => persons.first.mtSerie.timeLength;
//   int get variablesLength => persons.first.mtSerie.variablesLength;

//   @override
//   void paint(Canvas canvas, Size size) {
//     _canvas = canvas;
//     _width = size.width;
//     _height = size.height;

//     _blockWidth = _width / (timeLength * variablesLength);

//     plotGrid();
//   }

//   void plotGrid() {
//     for (var i = 0; i < persons.length; i++) {
//       plotPersonValues(persons[i], i);
//     }
//   }

//   void plotPersonValues(PersonModel personModel, int position) {
//     for (var i = 0; i < variablesLength; i++) {
//       for (var i = 0; i < timeLength; i++) {
//         _canvas.drawRect(Rect.fr, paint);
//       }
//     }
//   }

//   @override
//   bool shouldRepaint(CustomPainter oldDelegate) {
//     return true;
//   }
// }
