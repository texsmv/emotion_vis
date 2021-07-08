import 'package:emotion_vis_client/interfaces/ui_utils.dart';
import 'package:emotion_vis_client/models/dataset_settings.dart';
import 'package:emotion_vis_client/models/visualization_levels.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:menu_button/menu_button.dart';

typedef VisCallback = void Function(NonTemporalVisualization vis);

class NonTemporalPicker extends StatefulWidget {
  final NonTemporalVisualization initialVisualization;
  bool isTagged;
  int numberOfDimensions;
  ModelType type;
  final VisCallback onVisChanged;
  NonTemporalPicker({
    Key key,
    @required this.initialVisualization,
    @required this.onVisChanged,
  }) : super(key: key);

  @override
  _DisTemPickerState createState() => _DisTemPickerState();
}

class _DisTemPickerState extends State<NonTemporalPicker> {
  NonTemporalVisualization selectedVis;

  @override
  void initState() {
    selectedVis = widget.initialVisualization;
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      child: MenuButton<NonTemporalVisualization>(
        child: Container(
          width: 120,
          height: 30,
          color: Get.theme.primaryColor,
          alignment: Alignment.center,
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: Text(
            uiUtilNonTemVis2Str(selectedVis),
            textAlign: TextAlign.center,
            style: TextStyle(color: Colors.white, fontWeight: FontWeight.w400),
          ),
        ),
        items: uiUtilAvailableNonTemporalVisualizations(
            widget.type, widget.numberOfDimensions),
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
        itemBuilder: (NonTemporalVisualization visualization) => Container(
            width: 80,
            height: 30,
            color: Colors.white,
            alignment: Alignment.centerLeft,
            child: Text(uiUtilNonTemVis2Str(visualization))),
      ),
    );
  }
}
