import 'package:emotion_vis_client/interfaces/visualizations/non_temporal/categorical_scatterplot/categorical_scatterplot.dart';
import 'package:emotion_vis_client/interfaces/visualizations/non_temporal/dimensional_scatterplot/dimensional_scatterplot.dart';
import 'package:emotion_vis_client/interfaces/visualizations/non_temporal/glyph/glyph.dart';
import 'package:emotion_vis_client/interfaces/visualizations/non_temporal/glyph/glyph_single.dart';
import 'package:emotion_vis_client/interfaces/visualizations/non_temporal/polar_coord_line/polar_coord_line.dart';
import 'package:emotion_vis_client/interfaces/visualizations/temporal/linear_chart/linear_chart.dart';
import 'package:emotion_vis_client/interfaces/visualizations/temporal/stack_chart/stack_chart.dart';
import 'package:emotion_vis_client/interfaces/visualizations/temporal/temporal_glyph/temporal_glyph.dart';
import 'package:emotion_vis_client/interfaces/visualizations/vis_settings.dart';
import 'package:emotion_vis_client/models/dataset_settings.dart';
import 'package:emotion_vis_client/models/person_model.dart';
import 'package:emotion_vis_client/models/visualization_levels.dart';
import 'package:flutter/material.dart';

class NonTemporalChart extends StatelessWidget {
  final ModelType modelType;
  final int timePoint;
  final PersonModel personModel;
  // final DimInsVis dimInsVis;
  // final DisInsVis disInsVis;
  final NonTemporalVisualization nonTemporalVisualization;
  final VisSettings visSettings;
  const NonTemporalChart({
    Key key,
    @required this.modelType,
    @required this.personModel,
    // @required this.dimInsVis,
    // @required this.disInsVis,
    @required this.nonTemporalVisualization,
    @required this.visSettings,
    @required this.timePoint,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    Widget visualization;
    switch (nonTemporalVisualization) {
      case NonTemporalVisualization.CATEGORICAL_SCATTERPLOT:
        visualization = CategoricalScatterplot(
          personModel: personModel,
          visSettings: visSettings,
        );
        break;
      case NonTemporalVisualization.DIMENSIONAL_SCATTERPLOT:
        visualization = DimensionalScatterplot(
          personModel: personModel,
          visSettings: visSettings,
        );
        break;
      case NonTemporalVisualization.INSTANT_GLYPH:
        visualization = Glyph(
          personModel: personModel,
          visSettings: visSettings,
        );
        break;
      case NonTemporalVisualization.POLAR_LINES:
        visualization = PolarCoordLine(
          personModel: personModel,
          visSettings: visSettings,
        );
        break;
      case NonTemporalVisualization.INSTANT_GLYPH_SINGLE:
        visualization = GlyphSingle(
          personModel: personModel,
          visSettings: visSettings,
        );
        break;
      default:
    }
    return visualization;

    // Widget visualization;
    // if (modelType == ModelType.DIMENSIONAL) {
    //   switch (dimInsVis) {
    //     case DimInsVis.DimensionalScatterplot:
    //       visualization = DimensionalScatterplot(
    //         personModel: personModel,
    //         visSettings: visSettings,
    //       );
    //       break;
    //     case DimInsVis.GLYPH:
    //       visualization = Glyph(
    //         personModel: personModel,
    //         visSettings: visSettings,
    //       );
    //       break;
    //     default:
    //   }
    // } else {
    //   switch (disInsVis) {
    //     case DisInsVis.POLAR_COORD:
    //       visualization = PolarCoordLine(
    //         personModel: personModel,
    //         visSettings: visSettings,
    //       );
    //       break;
    //     case DisInsVis.CategoricalScatterplot:
    //       visualization = CategoricalScatterplot(
    //         personModel: personModel,
    //         visSettings: visSettings,
    //       );
    //       break;
    //     default:
    //       visualization = Container();
    //       break;
    //   }
    // }
    // return visualization;
  }
}
