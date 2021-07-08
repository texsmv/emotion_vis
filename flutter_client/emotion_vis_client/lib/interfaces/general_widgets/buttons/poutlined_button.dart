import 'package:emotion_vis_client/interfaces/constants/colors.dart';
import 'package:flutter/material.dart';

class POutlinedButton extends StatelessWidget {
  final String text;
  final VoidCallback onPressed;
  final Color buttonColor;
  const POutlinedButton({
    Key key,
    this.onPressed,
    this.text,
    this.buttonColor = pColorPrimary,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      // width: 100,
      child: OutlinedButton(
        style: OutlinedButton.styleFrom(
          backgroundColor: Colors.white,
          side: BorderSide(
            color: buttonColor,
          ),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(20.0),
          ),
        ),
        onPressed: onPressed,
        child: Text(
          text,
          style: TextStyle(
            color: buttonColor,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
    );
  }

  Color getColor(Set<MaterialState> states) {
    const Set<MaterialState> interactiveStates = <MaterialState>{
      MaterialState.pressed,
      MaterialState.hovered,
      MaterialState.focused,
      MaterialState.disabled,
    };
    if (states.any(interactiveStates.contains)) {
      return buttonColor;
    }
    return pColorError;
  }

  Color getColorBackground(Set<MaterialState> states) {
    const Set<MaterialState> interactiveStates = <MaterialState>{
      MaterialState.pressed,
      MaterialState.hovered,
      MaterialState.focused,
      MaterialState.disabled,
    };
    if (states.any(interactiveStates.contains)) {
      return buttonColor;
    }
    return pColorError;
  }
}
