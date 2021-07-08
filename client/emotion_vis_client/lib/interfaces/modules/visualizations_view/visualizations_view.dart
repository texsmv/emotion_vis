import 'package:emotion_vis_client/controllers/series_controller.dart';
import 'package:emotion_vis_client/interfaces/constants/colors.dart';
import 'package:emotion_vis_client/interfaces/general_widgets/containers/pcard.dart';
import 'package:emotion_vis_client/interfaces/general_widgets/others/emotions_legend.dart';
import 'package:emotion_vis_client/interfaces/modules/home/home_ui_controller.dart';
import 'package:emotion_vis_client/interfaces/modules/visualizations_view/components/all_view.dart';
import 'package:emotion_vis_client/interfaces/modules/visualizations_view/components/multiple_grid_view.dart';
import 'package:emotion_vis_client/interfaces/modules/visualizations_view/visualization_view_ui_controller.dart';
import 'package:emotion_vis_client/models/visualization_levels.dart';
import 'package:flutter/material.dart';
import 'package:flutter_icons/flutter_icons.dart';
import 'package:get/get.dart';
import 'package:menu_button/menu_button.dart';

class VisualizationsView extends StatefulWidget {
  const VisualizationsView({Key key}) : super(key: key);

  @override
  _VisualizationsViewState createState() => _VisualizationsViewState();
}

class _VisualizationsViewState extends State<VisualizationsView> {
  final HomeUiController _homeUiController = Get.find();

  @override
  void initState() {
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
      child: GetBuilder<HomeUiController>(
        builder: (_) => Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            EmotionsLegend(
              temporalVisualization: _homeUiController.temporalVisualization,
            ),
            const SizedBox(height: 10),
            const FilterBar(),
            const SizedBox(height: 5),
            // ignore: prefer_const_constructors
            ViewOptions(),
            const SizedBox(height: 10),
            Expanded(
              child: _getView(),
            ),
          ],
        ),
      ),
    );
  }

  Widget _getView() {
    switch (_homeUiController.selectedVisualization) {
      case MultipleView.GRID:
        return MultipleGridView();
      case MultipleView.LIST:
        return AllView();
      default:
        return null;
    }
  }
}

class ViewOptions extends StatelessWidget {
  const ViewOptions({Key key}) : super(key: key);

  HomeUiController get _homeUiController => Get.find();
  VisualizationsViewUiController get _viewController => Get.find();

  @override
  Widget build(BuildContext context) {
    return Row(
      children: List.generate(
        MultipleView.values.length,
        (index) => SizedBox(
          height: 50,
          child: Row(
            children: [
              TextButton(
                style: ButtonStyle(
                  shape: MaterialStateProperty.all<RoundedRectangleBorder>(
                    RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(0),
                      // side: BorderSide(color: pColorGray),
                    ),
                  ),
                  backgroundColor: MaterialStateProperty.all<Color>(
                    MultipleView.values[index] ==
                            _homeUiController.selectedVisualization
                        ? pColorPrimary
                        : Colors.white,
                  ),
                  foregroundColor:
                      MaterialStateProperty.all<Color>(Colors.black),
                ),
                onPressed: () {
                  _homeUiController.changeAllView(MultipleView.values[index]);
                },
                child: Text(
                  _multipleViewToStr(MultipleView.values[index]),
                  style: TextStyle(
                      color: MultipleView.values[index] ==
                              _homeUiController.selectedVisualization
                          ? const Color.fromARGB(255, 240, 240, 240)
                          : pTextColorPrimary),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  String _multipleViewToStr(MultipleView view) {
    switch (view) {
      case MultipleView.GRID:
        return "Grid";
      case MultipleView.LIST:
        return "List";
      default:
        return null;
    }
  }
}

class FilterBar extends GetView<VisualizationsViewUiController> {
  const FilterBar({Key key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return PCard(
      child: GetBuilder<SeriesController>(
        builder: (_) => GetBuilder<VisualizationsViewUiController>(
          builder: (_) => SizedBox(
            width: controller.isDataClustered ? 700 : 530,
            height: 40,
            child: Row(
              children: [
                const Icon(
                  Ionicons.ios_search,
                  size: 24,
                  color: pColorGray,
                ),
                Expanded(
                  child: TextField(
                    controller: controller.searchController,
                    textAlignVertical: TextAlignVertical.center,
                    decoration: const InputDecoration(
                      contentPadding: EdgeInsets.all(10.0),
                      alignLabelWithHint: true,
                      enabledBorder: OutlineInputBorder(
                        borderSide:
                            BorderSide(width: 0.0, color: Colors.transparent),
                        borderRadius: BorderRadius.all(
                          Radius.circular(8.0),
                        ),
                      ),
                      focusedBorder: OutlineInputBorder(
                        borderSide:
                            BorderSide(width: 0.0, color: Colors.transparent),
                        borderRadius: BorderRadius.all(
                          Radius.circular(8.0),
                        ),
                      ),
                      hintText: "Search by id or any other identifier",
                      labelStyle: TextStyle(
                        color: pColorDark,
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                ),
                IconButton(
                  icon: const Icon(
                    Icons.close,
                  ),
                  iconSize: 20,
                  onPressed: controller.resetSearch,
                ),
                Visibility(
                  visible: controller.isDataClustered,
                  child: Row(
                    children: [
                      const Padding(
                        padding: EdgeInsets.symmetric(vertical: 3),
                        child: VerticalDivider(
                          thickness: 3,
                        ),
                      ),
                      const Text(
                        "Filter:",
                        style: TextStyle(
                          fontSize: 17,
                          fontWeight: FontWeight.w500,
                          color: pTextColorSecondary,
                        ),
                      ),
                      const SizedBox(width: 20),
                      MenuButton<String>(
                        child: Container(
                          width: 150,
                          height: 40,
                          decoration: BoxDecoration(
                            color: controller.selectedCluster == noneCluster
                                ? Colors.transparent
                                : pColorAccent,
                            border: Border.all(
                              width: 2,
                              color: controller.selectedCluster == noneCluster
                                  ? pColorAccent
                                  : Colors.transparent,
                            ),
                          ),
                          alignment: Alignment.center,
                          child: Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Text(
                                controller.selectedCluster,
                                style: TextStyle(
                                  color:
                                      controller.selectedCluster == noneCluster
                                          ? pTextColorPrimary
                                          : pTextColorWhite,
                                  fontSize: 16,
                                  fontWeight: FontWeight.w400,
                                ),
                              ),
                              const SizedBox(width: 30),
                              Icon(
                                AntDesign.down,
                                size: 15,
                                color: controller.selectedCluster == noneCluster
                                    ? pTextColorPrimary
                                    : pTextColorWhite,
                              )
                            ],
                          ),
                        ),
                        items: controller.clustersOptions,
                        topDivider: true,
                        scrollPhysics: const AlwaysScrollableScrollPhysics(),
                        onItemSelected: controller.selectCluster,
                        onMenuButtonToggle: (isToggle) {},
                        decoration: BoxDecoration(
                          border: Border.all(color: Colors.white.withAlpha(0)),
                          borderRadius:
                              const BorderRadius.all(Radius.circular(3.0)),
                          color: Colors.white.withAlpha(0),
                        ),
                        divider: Container(
                          height: 1,
                          color: Colors.grey,
                        ),
                        toggledChild: Container(
                          height: 30,
                        ),
                        itemBuilder: (String value) => Container(
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: Colors.white,
                          ),
                          width: 150,
                          height: 40,
                          alignment: Alignment.center,
                          child: Text(value),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
