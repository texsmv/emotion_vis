import 'package:emotion_vis_client/interfaces/ui_utils.dart';
import 'package:emotion_vis_client/models/dataset_settings.dart';
import 'package:emotion_vis_client/models/visualization_levels.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:menu_button/menu_button.dart';

typedef VisCallback = void Function(TemporalVisualization vis);

class TemporalPicker extends StatefulWidget {
  final TemporalVisualization initialVisualization;
  bool isTagged;
  int numberOfDimensions;
  ModelType type;
  final VisCallback onVisChanged;
  TemporalPicker({
    Key key,
    @required this.initialVisualization,
    @required this.onVisChanged,
    @required this.isTagged,
    @required this.numberOfDimensions,
    @required this.type,
  }) : super(key: key);

  @override
  _DisTemPickerState createState() => _DisTemPickerState();
}

class _DisTemPickerState extends State<TemporalPicker> {
  TemporalVisualization selectedVis;

  @override
  void initState() {
    selectedVis = widget.initialVisualization;
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    // Container(
    //   height: 200,
    //   child: Column(
    //     children: [
    //       Text("Template"),
    //       ListView.builder(
    //         itemBuilder: (context, index) {
    //           return Container(
    //             height: 50,
    //             child:
    //                 Text(uiUtilTemVis2Str(TemporalVisualization.values[index])),
    //           );
    //         },
    //       )
    //     ],
    //   ),
    // );
    return Container(
      child: MenuButton<TemporalVisualization>(
        child: Container(
          width: 120,
          height: 30,
          color: Get.theme.primaryColor,
          alignment: Alignment.center,
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: Text(
            uiUtilTemVis2Str(selectedVis),
            textAlign: TextAlign.center,
            style: TextStyle(color: Colors.white, fontWeight: FontWeight.w400),
          ),
        ),
        items: uiUtilAvailableTemporalVisualizations(
            widget.isTagged, widget.type, widget.numberOfDimensions),
        topDivider: true,
        scrollPhysics: AlwaysScrollableScrollPhysics(),
        onItemSelected: (value) {
          selectedVis = value;
          widget.onVisChanged(value);
        },
        onMenuButtonToggle: (isToggle) {},
        decoration: BoxDecoration(
          border: Border.all(color: Colors.white.withAlpha(0)),
          borderRadius: const BorderRadius.all(Radius.circular(3.0)),
          color: Colors.white.withAlpha(0),
        ),
        divider: Container(
          height: 1,
          color: Colors.grey,
        ),
        toggledChild: Container(
          height: 30,
        ),
        itemBuilder: (TemporalVisualization visualization) => Container(
            width: 80,
            height: 30,
            color: Colors.white,
            alignment: Alignment.centerLeft,
            child: Text(uiUtilTemVis2Str(visualization))),
      ),
    );
  }
}
