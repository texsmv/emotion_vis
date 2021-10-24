import 'package:emotion_vis_client/interfaces/general_widgets/containers/pcard.dart';
import 'package:emotion_vis_client/interfaces/modules/projection_view/components/clusteredView.dart';
import 'package:emotion_vis_client/interfaces/modules/home/components/summary_visualization_view.dart';
import 'package:emotion_vis_client/interfaces/modules/projection_view/components/interactive_projection.dart';
import 'package:emotion_vis_client/interfaces/modules/projection_view/projection_view_ui_controller.dart';
import 'package:emotion_vis_client/interfaces/modules/projection_view/components/weights_selection.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';

/// This shows the projection view tab
class ProjectionView extends GetView<ProjectionViewUiController> {
  const ProjectionView({Key key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(20),
      clipBehavior: Clip.none,
      child: Row(
        children: [
          Expanded(
            flex: 7,
            child: Column(
              children: [
                // const ClusteringOptionsView(),
                const SizedBox(height: 20),
                const Expanded(
                  child: PCard(
                    padding: EdgeInsets.all(0),
                    child: InteractiveProjection(),
                  ),
                ),
                // const SizedBox(height: 20),
                // const WeigthsSelection(),
                // const SizedBox(height: 20),
                // const SummaryVisualizationView(),
              ],
            ),
          ),
          const SizedBox(width: 20),
          const Expanded(
            flex: 4,
            child: RepaintBoundary(
              child: ClusteredView(),
              // child: PCard(
              //   child: ClusteredView(),
              // ),
            ),
          ),
        ],
      ),
    );
  }
}
