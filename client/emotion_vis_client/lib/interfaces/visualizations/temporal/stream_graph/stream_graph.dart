import 'dart:math';

import 'package:charts_painter/chart.dart';
import 'package:emotion_vis_client/list_extension.dart';
import 'package:flutter/material.dart';

/// Stream graph of multivarite time series
///
/// The MTS must not have negative values
class StreamGraph extends StatefulWidget {
  /// Multivariate time series of shape T, D
  final List<dynamic> mts;
  final List<Color> colors;
  final Color backgroundColor;
  final List<String> timeLabels;
  const StreamGraph({
    Key key,
    @required this.mts,
    @required this.colors,
    this.backgroundColor = Colors.white,
    this.timeLabels,
  }) : super(key: key);

  @override
  _StreamGraphState createState() => _StreamGraphState();
}

class _StreamGraphState extends State<StreamGraph> {
  int get t => widget.mts.shape[1];
  int get d => widget.mts.shape[0];

  List<dynamic> get mts => widget.mts;

  List<double> _bottomPadding;

  List<Color> get _colors =>
      List.from(widget.colors)..add(widget.backgroundColor);

  List<List<BubbleValue<void>>> _bubbleValues;

  List<DecorationPainter> _backgroundDecorations;

  /// Multiplied for each value in the mts, so the labels can also be used
  /// for values lower than 1
  double _valueMultiplier = 1;

  bool _smoothPoints = true;

  @override
  void initState() {
    print("D: $d - T: $t");
    _updateValues();
    super.initState();
  }

  @override
  void didUpdateWidget(covariant StreamGraph oldWidget) {
    _updateValues();
    super.didUpdateWidget(oldWidget);
  }

  void _updateValues() {
    _fillPadding();
    _fillBubbleValues();
    _fillDecorations();
  }

  void _fillDecorations() {
    _backgroundDecorations = [
      GridDecoration(
        showHorizontalGrid: true,
        showVerticalGrid: false,
        showTopHorizontalValue: true,
        showVerticalValues: true,
        showHorizontalValues: true,
        horizontalAxisStep: 3,
        horizontalAxisValueFromValue: (int value) {
          return (value / _valueMultiplier).toStringAsFixed(2);
        },
        verticalAxisValueFromIndex: (index) {
          if (widget.timeLabels == null) {
            return index.toString();
          } else {
            return widget.timeLabels[index];
          }
        },
        textStyle: TextStyle(
          fontSize: 12,
          color: Colors.black,
        ),
        gridColor: Colors.black.withOpacity(0.2),
      )
    ];

    for (int i = 0; i < d + 1; i++) {
      _backgroundDecorations.add(
        SparkLineDecoration(
          id: '${i}_line_fill',
          smoothPoints: _smoothPoints,
          fill: true,
          lineColor: _colors[i],
          lineArrayIndex: i,
        ),
      );
    }
  }

  void _fillBubbleValues() {
    _bubbleValues = [];

    // Map the mts values
    for (var i = 0; i < d; i++) {
      final List<BubbleValue<void>> dValues = List.generate(t, (index) => null);
      for (var j = 0; j < t; j++) {
        double eps = 0;
        // if (j > 0) {
        //   if (mts[i][j] == mts[i][j - 1]) {
        //     eps = 0.001;
        //   }
        // }
        dValues[j] = BubbleValue<void>((mts[i][j] + eps) * _valueMultiplier);
      }
      _bubbleValues.add(dValues);
    }

    // Map the padding values
    final List<BubbleValue<void>> paddingValues =
        List.generate(t, (index) => null);
    for (var j = 0; j < t; j++) {
      paddingValues[j] =
          BubbleValue<void>(_bottomPadding[j] * _valueMultiplier);
    }
    _bubbleValues.add(paddingValues);
  }

  void _fillPadding() {
    var rng = new Random();

    _bottomPadding = List.generate(t, (index) => 0);

    // Get the max height along the y axis
    List<double> csums = List.generate(t, (index) => 0);
    double cmax = 0;
    for (var i = 0; i < t; i++) {
      for (var j = 0; j < d; j++) {
        mts[j][i] += rng.nextDouble() * 0.001;
        csums[i] += mts[j][i];
      }

      if (cmax < csums[i]) {
        cmax = csums[i];
      }
    }

    // Fills the padding based on the max height so each time
    // step will be centered
    for (var i = 0; i < t; i++) {
      // print("i: $i  csum: ${csums[i]}");
      // double eps = 0.001;
      // if (i > 0) {
      //   if (csums[i] == csums[i - 1]) {
      //     print("EQUAL");
      //     mts[0][i] += eps;
      //   }
      // }
      //   // if (mts[i][0] == mts[i][j - 1]) {
      //   //   eps = 0.001;
      //   // }
      // }
      // for (var j = 0; j < d; j++) {
      //   csums[i] += mts[j][i];
      // }
      _bottomPadding[i] = (cmax - csums[i]) / 2;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      color: Colors.white,
      child: LineChart<void>.multiple(
        _bubbleValues,
        stack: true,
        itemColor: Colors.transparent,
        lineWidth: 2.0,
        chartItemOptions: BubbleItemOptions(
          maxBarWidth: 6.0,
          colorForKey: (item, key) {
            return _colors[key];
          },
        ),
        smoothCurves: _smoothPoints,
        backgroundDecorations: _backgroundDecorations,
      ),
    );
  }
}

typedef DataToValue<T> = double Function(T item);
typedef DataToAxis<T> = String Function(int item);

class LineChart<T> extends StatelessWidget {
  LineChart({
    @required this.data,
    @required this.dataToValue,
    this.height = 240.0,
    this.lineWidth = 2.0,
    this.itemColor,
    this.backgroundDecorations,
    this.foregroundDecorations,
    this.chartItemOptions,
    this.chartBehaviour,
    this.smoothCurves,
    this.gradient,
    this.stack = false,
    Key key,
  })  : _mappedValues = [
          data.map((e) => BubbleValue<T>(dataToValue(e))).toList()
        ],
        super(key: key);

  LineChart.multiple(
    this._mappedValues, {
    this.height = 240.0,
    this.lineWidth = 2.0,
    this.itemColor,
    this.backgroundDecorations,
    this.foregroundDecorations,
    this.chartItemOptions,
    this.chartBehaviour,
    this.smoothCurves,
    this.gradient,
    this.stack = false,
    Key key,
  })  : data = null,
        dataToValue = null,
        super(key: key);

  final List<T> data;
  final DataToValue<T> dataToValue;

  final double height;

  final bool smoothCurves;
  final Color itemColor;
  final Gradient gradient;
  final double lineWidth;
  final bool stack;

  final List<DecorationPainter> backgroundDecorations;
  final List<DecorationPainter> foregroundDecorations;
  final ChartBehaviour chartBehaviour;
  final ItemOptions chartItemOptions;

  final List<List<ChartItem<T>>> _mappedValues;

  @override
  Widget build(BuildContext context) {
    final _foregroundDecorations =
        foregroundDecorations ?? <DecorationPainter>[];
    final _backgroundDecorations =
        backgroundDecorations ?? <DecorationPainter>[];

    return AnimatedChart<T>(
      height: height,
      duration: const Duration(milliseconds: 450),
      state: ChartState<T>(
        ChartData(_mappedValues,
            strategy: stack ? DataStrategy.stack : DataStrategy.none),
        itemOptions: chartItemOptions,
        foregroundDecorations: [
          SparkLineDecoration(
            id: 'chart_decoration',
            lineWidth: lineWidth,
            lineColor: itemColor,
            gradient: gradient,
            smoothPoints: smoothCurves,
          ),
          ..._foregroundDecorations,
        ],
        backgroundDecorations: [
          ..._backgroundDecorations,
        ],
      ),
    );
  }
}
