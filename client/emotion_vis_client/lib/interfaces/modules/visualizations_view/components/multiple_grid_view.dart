import 'package:auto_size_text/auto_size_text.dart';
import 'package:emotion_vis_client/controllers/series_controller.dart';
import 'package:emotion_vis_client/interfaces/modules/home/components/summary_visualization_view.dart';
import 'package:emotion_vis_client/interfaces/modules/visualizations_view/visualization_view_ui_controller.dart';
import 'package:emotion_vis_client/interfaces/ui_utils.dart';
import 'package:emotion_vis_client/models/person_model.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:rainbow_color/rainbow_color.dart';

class MultipleGridView extends StatefulWidget {
  MultipleGridView({Key key}) : super(key: key);

  @override
  _MultipleGridViewState createState() => _MultipleGridViewState();
}

class _MultipleGridViewState extends State<MultipleGridView> {
  VisualizationsViewUiController controller = Get.find();

  List<PersonModel> get persons => controller.persons;

  double _width;
  double _height;
  double _blockSize;
  double _separationPercentage = 0.05;
  double _textSpace = 100;
  int get timeLength => persons.first.mtSerie.timeLength;
  int get variablesLength => controller.datasetSettings.variablesNames.length;
  Map<String, Rainbow> colorInterpolators;
  List<String> get variables => controller.datasetSettings.variablesNames;

  @override
  void initState() {
    colorInterpolators = {};
    for (var i = 0; i < variablesLength; i++) {
      colorInterpolators[variables[i]] = Rainbow(spectrum: [
        Colors.white,
        controller.datasetSettings.variablesColors[variables[i]]
      ], rangeStart: 0, rangeEnd: 1);
    }
    WidgetsBinding.instance.addPostFrameCallback((_) => setState(() {}));
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Expanded(
          child: GetBuilder<SeriesController>(
            builder: (_) => Column(
              children: [
                Row(
                  children: [
                    Container(
                      height: _textSpace,
                      width: _textSpace,
                    ),
                    Expanded(
                      child: Container(
                        width: double.infinity,
                        height: _textSpace,
                        child: Row(
                          children: List.generate(
                            variablesLength,
                            (index) => Expanded(
                              child: Row(
                                mainAxisAlignment:
                                    MainAxisAlignment.spaceAround,
                                children: timeLabels(),
                              ),
                            ),
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
                Expanded(
                  child: LayoutBuilder(
                    builder: (context, constraints) {
                      _width = constraints.maxWidth;
                      _height = constraints.maxHeight;
                      _blockSize = ((_width - _textSpace) /
                              (timeLength * variablesLength)) *
                          (1 - _separationPercentage);
                      return Container(
                        child: ListView.separated(
                          separatorBuilder: (context, index) => SizedBox(
                              height: _separationPercentage * _blockSize),
                          itemBuilder: (context, index) {
                            return Row(
                              mainAxisAlignment: MainAxisAlignment.end,
                              children: [
                                Container(
                                  padding: EdgeInsets.symmetric(horizontal: 5),
                                  width: _textSpace,
                                  child: Text(
                                    persons[index].id,
                                    textAlign: TextAlign.end,
                                  ),
                                ),
                                Expanded(child: personValues(persons[index])),
                              ],
                            );
                          },
                          itemCount: persons.length,
                        ),
                      );
                    },
                  ),
                ),
              ],
            ),
          ),
        ),
        SizedBox(height: 20),
        const SummaryVisualizationView(),
      ],
    );
  }

  List<Widget> timeLabels() {
    List<Widget> children = [];
    for (var i = 0; i < timeLength; i++) {
      children.add(RotatedBox(
        quarterTurns: 3,
        child: SizedBox(
          height: _blockSize,
          child: AutoSizeText(
            controller.datasetSettings.labels[i],
            style: TextStyle(fontSize: 15),
            maxLines: 1,
            minFontSize: 5,
          ),
        ),
      ));
    }
    return children;
  }

  Widget personValues(PersonModel personModel) {
    List<Widget> children = [];
    for (var i = 0; i < variablesLength; i++) {
      String variable = variables[i];
      final double minValue = controller.datasetSettings.minValues[variable];
      final double maxValue = controller.datasetSettings.maxValues[variable];

      for (var j = 0; j < timeLength; j++) {
        double scaledValue = uiUtilRangeConverter(
            personModel.mtSerie.at(j, variables[i]), minValue, maxValue, 0, 1);
        children.add(
          Container(
            width: _blockSize,
            height: _blockSize,
            decoration: BoxDecoration(
              color: colorInterpolators[variable][scaledValue],
              borderRadius: BorderRadius.circular(_blockSize / 4),
            ),
          ),
        );
      }
    }
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceAround,
      children: children,
    );
  }
}
