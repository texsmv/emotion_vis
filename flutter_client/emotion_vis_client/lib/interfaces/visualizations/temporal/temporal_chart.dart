import 'package:emotion_vis_client/interfaces/visualizations/temporal/linear_chart/linear_chart.dart';
import 'package:emotion_vis_client/interfaces/visualizations/temporal/polar_bars/polar_bars.dart';
import 'package:emotion_vis_client/interfaces/visualizations/temporal/stack_chart/stack_chart.dart';
import 'package:emotion_vis_client/interfaces/visualizations/temporal/tagged_tunnel/tagged_tunnel.dart';
import 'package:emotion_vis_client/interfaces/visualizations/temporal/temporal_glyph/temporal_glyph.dart';
import 'package:emotion_vis_client/interfaces/visualizations/temporal/temporal_tunnel/temporal_tunnel.dart';
import 'package:emotion_vis_client/interfaces/visualizations/vis_settings.dart';
import 'package:emotion_vis_client/models/dataset_settings.dart';
import 'package:emotion_vis_client/models/person_model.dart';
import 'package:emotion_vis_client/models/visualization_levels.dart';
import 'package:flutter/material.dart';

class TemporalChart extends StatefulWidget {
  final ModelType modelType;
  final PersonModel personModel;
  final VisSettings visSettings;
  final TemporalVisualization temporalVisualization;
  const TemporalChart({
    Key key,
    @required this.modelType,
    @required this.personModel,
    @required this.temporalVisualization,
    @required this.visSettings,
  }) : super(key: key);

  @override
  _TemporalChartState createState() => _TemporalChartState();
}

class _TemporalChartState extends State<TemporalChart> {
  PersonModel get personModel => widget.personModel;
  VisSettings get visSettings => widget.visSettings;
  @override
  Widget build(BuildContext context) {
    if (!personModel.isDataLoaded) {
      personModel.loadEmotions().then((value) {
        setState(() {});
      });
      return const SizedBox(
        child: Center(
          child: CircularProgressIndicator(),
        ),
      );
    }

    Widget visualization;
    switch (widget.temporalVisualization) {
      case TemporalVisualization.LINEAR_CHART:
        visualization = TemporalLinearChart(
          personModel: personModel,
          visSettings: visSettings,
        );
        break;
      case TemporalVisualization.STACKED_CHART:
        visualization = StackChart(
          personModel: personModel,
          visSettings: visSettings,
        );
        break;
      case TemporalVisualization.POLAR_BARS:
        visualization = PolarBars(
          personModel: personModel,
          visSettings: visSettings,
        );
        break;
      case TemporalVisualization.TAGGED_TUNNEL:
        visualization = TaggedTunnel(
          personModel: personModel,
          visSettings: visSettings,
        );
        break;
      case TemporalVisualization.TEMPORAL_GLYPH:
        visualization = TemporalGlyph(
          personModel: personModel,
          visSettings: visSettings,
        );
        break;
      case TemporalVisualization.TEMPORAL_TUNNEL:
        visualization = TemporalTunnel(
          personModel: personModel,
          visSettings: visSettings,
        );
        break;
      default:
    }
    return visualization;
  }
}
