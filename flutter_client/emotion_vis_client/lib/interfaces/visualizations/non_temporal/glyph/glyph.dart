import 'dart:math';
import 'dart:ui';

import 'package:emotion_vis_client/interfaces/ui_utils.dart';
import 'package:emotion_vis_client/interfaces/visualizations/vis_settings.dart';
import 'package:emotion_vis_client/models/person_model.dart';
import 'package:flutter/material.dart';

class Glyph extends StatefulWidget {
  double radius = 140;
  PersonModel personModel;
  VisSettings visSettings;
  Glyph({
    Key key,
    @required this.personModel,
    @required this.visSettings,
  }) : super(key: key);

  @override
  _GlyphState createState() => _GlyphState();
}

class _GlyphState extends State<Glyph> with TickerProviderStateMixin {
  AnimationController _valenceController;
  AnimationController _arousalController;

  double currentRadius;
  double get neutralRadius => widget.radius * 0.8;
  bool valenceNeutral;
  @override
  void initState() {
    super.initState();
    _valenceController = AnimationController(
      duration: const Duration(seconds: 10),
      vsync: this,
    )..repeat(reverse: false);

    _arousalController = AnimationController(
      duration: const Duration(seconds: 2),
      vsync: this,
    )..repeat(reverse: false);
    currentRadius = neutralRadius;
  }

  @override
  void dispose() {
    _valenceController.dispose();
    _arousalController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      width: widget.radius * 2,
      height: widget.radius * 2,
      child: Stack(
        children: [
          Container(
            alignment: Alignment.center,
            child: RotationTransition(
              turns: _valenceController,
              child: _valenceWidget(),
            ),
          ),
          Positioned.fill(
            child: Container(
              height: double.infinity,
              width: double.infinity,
              alignment: Alignment.center,
              child: _arousalWidget(),
            ),
          ),
          Positioned.fill(
            child: Center(
              child: _dominanceWidget(),
            ),
          )
        ],
      ),
    );
  }

  Widget _dominanceWidget() {
    if (widget.visSettings.variablesNames.length == 2) return Container();
    double dominanceValue = uiUtilRangeConverter(
        widget.personModel.mtSerie
            .at(widget.visSettings.timePoint, widget.visSettings.dominance),
        widget.visSettings.datasetSettings
            .minValues[widget.visSettings.dominance],
        widget.visSettings.datasetSettings
            .maxValues[widget.visSettings.dominance],
        0,
        1);
    if (dominanceValue < 0.5) {
      if (dominanceValue < 0.25) {
        if (dominanceValue < 0.125) {
          currentRadius = widget.radius * 0.6;
        } else {
          currentRadius = widget.radius * 0.7;
        }
      } else {
        if (dominanceValue < 0.375) {
          currentRadius = widget.radius * 0.7;
        } else {
          currentRadius = widget.radius * 0.8;
        }
      }
    } else {
      if (dominanceValue < 0.75) {
        if (dominanceValue < 0.625) {
          currentRadius = widget.radius * 0.8;
        } else {
          currentRadius = widget.radius * 0.9;
        }
      } else {
        if (dominanceValue < 0.875) {
          currentRadius = widget.radius * 0.9;
        } else {
          currentRadius = widget.radius * 1;
        }
      }
    }
    return Container(
      height: neutralRadius * 2,
      width: neutralRadius * 2,
      alignment: Alignment.center,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        border: Border.all(width: 1, color: Colors.black),
      ),
    );
  }

  Widget _valenceWidget() {
    double valenceValue = uiUtilRangeConverter(
        widget.personModel.mtSerie
            .at(widget.visSettings.timePoint, widget.visSettings.valence),
        widget
            .visSettings.datasetSettings.minValues[widget.visSettings.valence],
        widget
            .visSettings.datasetSettings.maxValues[widget.visSettings.valence],
        0,
        1);
    String asset;
    if (valenceValue < 0.5) {
      if (valenceValue < 0.25) {
        if (valenceValue < 0.125) {
          asset = "assets/glyph_1.png";
          valenceNeutral = false;
        } else {
          asset = "assets/glyph_2.png";
          valenceNeutral = false;
        }
      } else {
        if (valenceValue < 0.375) {
          asset = "assets/glyph_2.png";
          valenceNeutral = false;
        } else {
          asset = "assets/glyph_3.png";
          valenceNeutral = true;
        }
      }
    } else {
      if (valenceValue < 0.75) {
        if (valenceValue < 0.625) {
          asset = "assets/glyph_3.png";
          valenceNeutral = true;
        } else {
          asset = "assets/glyph_4.png";
          valenceNeutral = false;
        }
      } else {
        if (valenceValue < 0.875) {
          asset = "assets/glyph_4.png";
          valenceNeutral = false;
        } else {
          asset = "assets/glyph_5.png";
          valenceNeutral = false;
        }
      }
    }
    double size = currentRadius * 2 * 1.1;

    if (valenceNeutral) {
      size = currentRadius * 2;
    }

    return ImageIcon(
      AssetImage(asset),
      size: size,
      color: widget.visSettings.colors[widget.visSettings.arousal],
    );
  }

  Widget _arousalWidget() {
    double arousalValue = uiUtilRangeConverter(
        widget.personModel.mtSerie
            .at(widget.visSettings.timePoint, widget.visSettings.arousal),
        widget
            .visSettings.datasetSettings.minValues[widget.visSettings.arousal],
        widget
            .visSettings.datasetSettings.maxValues[widget.visSettings.arousal],
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
      animation: _arousalController,
      builder: (context, child) => Container(
        width: double.infinity,
        height: double.infinity,
        child: CustomPaint(
          painter: CircleBlurPainter(
            circleWidth: circleWidth,
            blurSigma: blurSigma,
            maxRadius: currentRadius,
            value: _arousalController.value,
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
