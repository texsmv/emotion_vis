import 'package:emotion_vis_client/interfaces/constants/colors.dart';
import 'package:flutter/material.dart';

class PCard extends StatelessWidget {
  final Widget child;
  final EdgeInsets padding;
  final double borderRadius;
  const PCard({
    Key key,
    @required this.child,
    this.padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
    this.borderRadius = 6,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: pColorBackground,
        borderRadius: BorderRadius.circular(borderRadius),
      ),
      padding: padding,
      child: child,
    );
  }
}
