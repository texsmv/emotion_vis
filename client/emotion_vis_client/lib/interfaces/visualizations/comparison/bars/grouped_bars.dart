import 'package:charts_painter/chart.dart';
import 'package:emotion_vis_client/controllers/series_controller.dart';
import 'package:emotion_vis_client/models/dataset_settings.dart';
import 'package:emotion_vis_client/models/person_model.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';

class GroupedBars extends StatefulWidget {
  // final List<PersonModel> clusterA;
  // final List<PersonModel> clusterB;
  List<double> averagesA;
  List<double> averagesB;
  final Color colorA;
  final Color colorB;
  GroupedBars({
    Key key,
    // @required this.clusterA,
    // @required this.clusterB,
    @required this.averagesA,
    @required this.averagesB,
    @required this.colorA,
    @required this.colorB,
  }) : super(key: key);

  @override
  _GroupedBarsState createState() => _GroupedBarsState();
}

class _GroupedBarsState extends State<GroupedBars> {
  final SeriesController _seriesController = Get.find();
  DatasetSettings get datasetSettings => _seriesController.datasetSettings;
  List<String> get labels => datasetSettings.variablesNames;
  List<double> get averagesA => widget.averagesA;
  List<double> get averagesB => widget.averagesB;
  // int get timeLen => widget.clusterA[0].mtSerie.timeLength;

  @override
  void initState() {
    // getClusterAaverages();
    // getClusterBaverages();
    super.initState();
  }

  @override
  void didUpdateWidget(covariant GroupedBars oldWidget) {
    // getClusterAaverages();
    // getClusterBaverages();
    super.didUpdateWidget(oldWidget);
  }

  // void getClusterAaverages() {
  //   averagesA = List.generate(labels.length, (index) => 0);
  //   for (var i = 0; i < labels.length; i++) {
  //     for (var k = 0; k < widget.clusterA.length; k++) {
  //       final PersonModel person = widget.clusterA[k];
  //       for (var j = 0; j < timeLen; j++) {
  //         averagesA[i] = averagesA[i] + person.mtSerie.at(j, labels[i]);
  //       }
  //     }
  //     averagesA[i] = averagesA[i] / (widget.clusterA.length * timeLen);
  //   }
  // }

  // void getClusterBaverages() {
  //   averagesB = List.generate(labels.length, (index) => 0);
  //   for (var i = 0; i < labels.length; i++) {
  //     for (var k = 0; k < widget.clusterB.length; k++) {
  //       final PersonModel person = widget.clusterB[k];
  //       for (var j = 0; j < timeLen; j++) {
  //         averagesB[i] = averagesB[i] + person.mtSerie.at(j, labels[i]);
  //       }
  //     }
  //     averagesB[i] = averagesB[i] / (widget.clusterB.length * timeLen);
  //   }
  // }

  @override
  Widget build(BuildContext context) {
    return AnimatedChart(
      duration: const Duration(milliseconds: 500),
      state: ChartState(
        ChartData(
          [
            averagesA.map((e) => BarValue<void>(e * 10)).toList(),
            averagesB.map((e) => BarValue<void>(e * 10)).toList()
          ],
          axisMax: datasetSettings.maxValue * 10,
          axisMin: datasetSettings.minValue * 10,
        ),
        behaviour: const ChartBehaviour(
          multiItemStack: false,
        ),
        itemOptions: BarItemOptions(
          maxBarWidth: 30,
          minBarWidth: 4,
          colorForKey: (item, index) {
            return [widget.colorA, widget.colorB][index];
          },
        ),
        foregroundDecorations: [
          // SparkLineDecoration(
          //   lineWidth: 4.0,
          //   lineColor: Theme.of(context).colorScheme.primary,
          // ),
          // BorderDecoration(
          //   color: Theme.of(context).colorScheme.secondary,
          //   // width: 2.0,
          // ),
          // SparkLineDecoration(
          //   // Specify key that this [SparkLineDecoration] will follow
          //   // Throws if `lineKey` does not exist in chart data
          //   lineKey: 1,
          //   lineColor: Theme.of(context).colorScheme.primaryVariant,
          // ),
        ],
        backgroundDecorations: [
          GridDecoration(
            showVerticalGrid: false,
            // horizontalAxisStep: 1,
            gridColor: Theme.of(context).dividerColor,
            showHorizontalValues: true,
            showVerticalValues: true,
            horizontalAxisStep:
                (datasetSettings.maxValue - datasetSettings.minValue) / 5 * 10,
            horizontalAxisValueFromValue: (value) =>
                (value / 10.0).toStringAsFixed(1),
            verticalAxisValueFromIndex: (index) => labels[index],
            textStyle: TextStyle(
              fontSize: 13,
              color: Colors.black,
            ),
          ),
        ],
      ),
    );
  }
}
