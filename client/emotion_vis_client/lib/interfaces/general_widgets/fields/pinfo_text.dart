import 'package:emotion_vis_client/interfaces/constants/colors.dart';
import 'package:flutter/material.dart';

class PInfoText extends StatelessWidget {
  final String label;
  final String text;

  const PInfoText({
    Key key,
    @required this.label,
    @required this.text,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 50,
      width: double.infinity,
      decoration: BoxDecoration(
        border: Border.all(
          color: const Color.fromARGB(255, 240, 240, 240),
        ),
        borderRadius: BorderRadius.circular(8),
      ),
      padding: const EdgeInsets.symmetric(horizontal: 10),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const SizedBox(height: 4),
          Text(
            label,
            style: const TextStyle(
              color: pTextColorPrimary,
              fontWeight: FontWeight.w600,
              fontSize: 12,
            ),
          ),
          Expanded(
            child: Text(
              text,
              style: const TextStyle(
                color: pTextColorPrimary,
                fontSize: 14,
                fontWeight: FontWeight.w400,
              ),
            ),
          ),
          const SizedBox(height: 10),
        ],
      ),
    );
  }
}
