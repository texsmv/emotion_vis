import 'package:emotion_vis_client/interfaces/constants/colors.dart';
import 'package:flutter/material.dart';

class PFilledButton extends StatelessWidget {
  final String text;
  final VoidCallback onPressed;
  final Color buttonColor;
  final Color textColor;
  const PFilledButton(
      {Key key,
      this.onPressed,
      this.text,
      this.buttonColor = pColorPrimary,
      this.textColor = Colors.white})
      : super(key: key);

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      child: TextButton(
        style: ButtonStyle(
          foregroundColor: MaterialStateProperty.resolveWith(getColor),
          backgroundColor:
              MaterialStateProperty.resolveWith(getColorBackground),
          shape: MaterialStateProperty.all<RoundedRectangleBorder>(
            RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(20.0),
            ),
          ),
        ),
        onPressed: onPressed,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 8),
          child: Text(
            text,
            style: TextStyle(
              color: textColor,
              fontWeight: FontWeight.w600,
            ),
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
      return textColor;
    }
    return textColor;
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
    return buttonColor;
  }
}
