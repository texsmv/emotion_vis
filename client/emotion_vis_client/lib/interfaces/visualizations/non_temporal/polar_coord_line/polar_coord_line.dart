import 'package:emotion_vis_client/models/person_model.dart';
import 'package:flutter/material.dart';
import 'package:graphic/graphic.dart' as graphic;

import '../../vis_settings.dart';

class PolarCoordLine extends StatefulWidget {
  PersonModel personModel;
  VisSettings visSettings;
  PolarCoordLine({
    Key key,
    @required this.personModel,
    @required this.visSettings,
  }) : super(key: key);

  @override
  _PolarCoordLineState createState() => _PolarCoordLineState();
}

class _PolarCoordLineState extends State<PolarCoordLine> {
  List<Map<String, Object>> data = [];
  Widget chart;

  @override
  void initState() {
    setData();
    super.initState();
  }

  void setData() {
    data.clear();
    for (var i = 0; i < widget.visSettings.variablesNames.length; i++) {
      Map<String, Object> pointMap = {};
      pointMap["emotionDimension"] = "fixed";
      pointMap["label"] = widget.visSettings.variablesNames[i];
      pointMap["value"] = widget.personModel.mtSerie.at(
          widget.visSettings.timePoint, widget.visSettings.variablesNames[i]);
      data.add(pointMap);
    }

    chart = Container(
      key: UniqueKey(),
      child: graphic.Chart(
        data: data,
        scales: {
          'emotionDimension': graphic.CatScale(
            accessor: (map) => map['emotionDimension'] as String,
          ),
          'label': graphic.CatScale(
            accessor: (map) => map['label'] as String,
          ),
          'value': graphic.LinearScale(
            accessor: (map) => map['value'] as num,
            min: widget.visSettings.lowerLimit,
            max: widget.visSettings.upperLimit,
          )
        },
        coord: graphic.PolarCoord(),
        geoms: [
          graphic.LineGeom(
            position: graphic.PositionAttr(field: 'label*value'),
            color: graphic.ColorAttr(field: 'emotionDimension'),
          )
        ],
        axes: {
          'label': graphic.Defaults.circularAxis,
          'value': graphic.Defaults.radialAxis,
        },
      ),
    );
  }

  @override
  void didUpdateWidget(covariant PolarCoordLine oldWidget) {
    setData();
    super.didUpdateWidget(oldWidget);
  }

  @override
  Widget build(BuildContext context) {
    return chart;
  }
}
