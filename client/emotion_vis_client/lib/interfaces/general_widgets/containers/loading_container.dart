import 'package:emotion_vis_client/interfaces/constants/colors.dart';
import 'package:flutter/material.dart';
import 'package:glass_kit/glass_kit.dart';

class LoadingContainer extends StatelessWidget {
  final Widget child;
  final bool isLoading;
  const LoadingContainer({
    Key key,
    @required this.child,
    @required this.isLoading,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        child,
        if (isLoading)
          Positioned.fill(
            child: Container(
              width: double.infinity,
              height: double.infinity,
              color: pColorPrimary.withOpacity(0.3),
              child: const Center(
                child: CircularProgressIndicator(),
              ),
            ),
          )
        else
          const SizedBox(),
      ],
    );
  }
}
