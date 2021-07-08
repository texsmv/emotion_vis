import 'dart:math';
import 'dart:ui';

import 'package:emotion_vis_client/interfaces/ui_utils.dart';
import 'package:emotion_vis_client/interfaces/visualizations/vis_settings.dart';
import 'package:emotion_vis_client/models/person_model.dart';
import 'package:flutter/material.dart';

class GlyphSingle extends StatefulWidget {
  double radius = 140;
  PersonModel personModel;
  VisSettings visSettings;
  GlyphSingle({
    Key key,
    @required this.personModel,
    @required this.visSettings,
  }) : super(key: key);

  @override
  _GlyphSingleState createState() => _GlyphSingleState();
}

class _GlyphSingleState extends State<GlyphSingle>
    with TickerProviderStateMixin {
  AnimationController _controller;

  double currentRadius;
  double get neutralRadius => widget.radius * 0.8;
  bool valenceNeutral;
  @override
  void initState() {
    super.initState();
    // _valenceController = AnimationController(
    //   duration: const Duration(seconds: 10),
    //   vsync: this,
    // )..repeat(reverse: false);

    _controller = AnimationController(
      duration: const Duration(seconds: 2),
      vsync: this,
    )..repeat(reverse: false);
    currentRadius = neutralRadius;
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      width: widget.radius * 2,
      height: widget.radius * 2,
      alignment: Alignment.center,
      child: _arousalWidget(),
    );
  }

  Widget _arousalWidget() {
    double arousalValue = uiUtilRangeConverter(
        widget.personModel.mtSerie.at(widget.visSettings.timePoint,
            widget.visSettings.datasetSettings.variablesNames[0]),
        widget.visSettings.datasetSettings.minValue,
        widget.visSettings.datasetSettings.maxValue,
        0,
        1);
    int waveNumber = 1;
    double blurSigma = 1;
    double circleWidth = 1;

    if (arousalValue < 0.5) {
      if (arousalValue < 0.25) {
        if (arousalValue < 0.125) {
          waveNumber = 2;
          blurSigma = 13;
          circleWidth = 13;
        } else {
          waveNumber = 4;
          blurSigma = 8;
          circleWidth = 8;
        }
      } else {
        if (arousalValue < 0.375) {
          waveNumber = 4;
          blurSigma = 8;
          circleWidth = 8;
        } else {
          waveNumber = 6;
          blurSigma = 4;
          circleWidth = 4;
        }
      }
    } else {
      if (arousalValue < 0.75) {
        if (arousalValue < 0.625) {
          waveNumber = 6;
          blurSigma = 4;
          circleWidth = 4;
        } else {
          waveNumber = 9;
          blurSigma = 2;
          circleWidth = 2;
        }
      } else {
        if (arousalValue < 0.875) {
          waveNumber = 9;
          blurSigma = 2;
          circleWidth = 3;
        } else {
          waveNumber = 12;
          blurSigma = 1;
          circleWidth = 2;
        }
      }
    }
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) => Container(
        decoration: BoxDecoration(
          color: Colors.blue,
          shape: BoxShape.circle,
        ),
        width: currentRadius * 2,
        height: currentRadius * 2,
        child: CustomPaint(
          painter: CircleBlurPainter(
            circleWidth: circleWidth,
            blurSigma: blurSigma,
            maxRadius: currentRadius,
            value: _controller.value,
            waveNumber: waveNumber,
          ),
        ),
      ),
    );
  }
}

class CircleBlurPainter extends CustomPainter {
  AnimationController controller;
  double maxRadius;
  double value;
  int waveNumber;
  CircleBlurPainter({
    @required this.circleWidth,
    @required this.waveNumber,
    this.blurSigma = 1,
    @required this.maxRadius,
    @required this.value,
  });
  double waveGap;

  double circleWidth;
  double blurSigma;

  @override
  void paint(Canvas canvas, Size size) {
    Paint line = new Paint()
      ..color = Colors.white
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke
      ..strokeWidth = circleWidth
      ..maskFilter = MaskFilter.blur(BlurStyle.normal, blurSigma);
    Offset center = new Offset(size.width / 2, size.height / 2);
    double radius = min(size.width / 2, size.height / 2);

    waveGap = radius / waveNumber;
    double currentRadius = waveGap * value;
    while (currentRadius < maxRadius) {
      canvas.drawCircle(center, currentRadius, line);
      currentRadius += waveGap;
    }
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) {
    return true;
  }
}
