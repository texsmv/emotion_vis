import 'package:emotion_vis_client/interfaces/constants/colors.dart';
import 'package:flutter/material.dart';

class PTextField extends StatelessWidget {
  final Color color;
  final String label;
  final TextEditingController controller;
  final int maxLenght;
  const PTextField({
    Key key,
    this.color = pColorPrimary,
    this.controller,
    this.maxLenght,
    this.label,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return TextField(
      controller: controller,
      expands: true,
      maxLength: maxLenght,
      maxLines: null,
      textAlignVertical: TextAlignVertical.top,
      decoration: InputDecoration(
        alignLabelWithHint: true,
        enabledBorder: const OutlineInputBorder(
          borderSide: BorderSide(color: Colors.grey, width: 0.0),
          borderRadius: BorderRadius.all(
            Radius.circular(8.0),
          ),
        ),
        focusedBorder: OutlineInputBorder(
          borderSide: BorderSide(color: color, width: 0.0),
          borderRadius: const BorderRadius.all(
            Radius.circular(8.0),
          ),
        ),
        labelText: label,
        labelStyle: TextStyle(
          color: color,
          fontSize: 14,
          fontWeight: FontWeight.w400,
        ),
        floatingLabelBehavior: FloatingLabelBehavior.always,
      ),
    );
  }
}
