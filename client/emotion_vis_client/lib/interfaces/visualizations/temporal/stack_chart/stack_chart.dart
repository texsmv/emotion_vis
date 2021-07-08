import 'dart:ui';

import 'package:emotion_vis_client/models/MTSerie.dart';
import 'package:emotion_vis_client/models/dataset_settings.dart';
import 'package:emotion_vis_client/models/person_model.dart';
import 'package:flutter/material.dart';
import 'package:syncfusion_flutter_charts/charts.dart';
// import 'package:graphic/graphic.dart' as graphic;

import '../../vis_settings.dart';

class StackChart extends StatefulWidget {
  PersonModel personModel;
  VisSettings visSettings;
  StackChart({
    Key key,
    @required this.personModel,
    @required this.visSettings,
  }) : super(key: key);

  @override
  _StackChartState createState() => _StackChartState();
}

class _StackChartState extends State<StackChart> {
  List<List<EmotionData>> series = [];
  Widget chart;
  List<String> get variables => widget.visSettings.variablesNames;
  MTSerie get mtserie => widget.personModel.mtSerie;
  DatasetSettings get datasetSettings => widget.visSettings.datasetSettings;
  bool useAllLabels;

  @override
  void initState() {
    useAllLabels = mtserie.timeLength == datasetSettings.allLabels.length;
    setData();

    super.initState();
  }

  void setData() {
    series.clear();
    for (var i = 0; i < variables.length; i++) {
      List<EmotionData> serie = [];
      for (var j = 0; j < mtserie.timeLength; j++) {
        serie.add(EmotionData(
            label: useAllLabels
                ? datasetSettings.allLabels[j]
                : datasetSettings.labels[j],
            value:
                mtserie.at(j, variables[i]) - widget.visSettings.lowerLimit));
        
      }
      series.add(serie);
    }
  }

  @override
  void didUpdateWidget(covariant StackChart oldWidget) {
    setData();
    super.didUpdateWidget(oldWidget);
  }

  @override
  Widget build(BuildContext context) {
    return SfCartesianChart(
      primaryXAxis: CategoryAxis(),
      // title: ChartTitle(text: 'Half yearly sales analysis'),
      tooltipBehavior: TooltipBehavior(enable: true),
      series: List.generate(
        variables.length,
        (index) => StackedAreaSeries<EmotionData, String>(
          dataSource: series[index],
          xValueMapper: (EmotionData emotion, _) => emotion.label,
          yValueMapper: (EmotionData emotion, _) => emotion.value,
          name: variables[index],
          color: widget.visSettings.colors[variables[index]],
          // dataLabelSettings: DataLabelSettings(isVisible: true),
        ),
      ),
    );
  }
}

class EmotionData {
  final String label;
  final double value;
  EmotionData({this.label, this.value});
}
