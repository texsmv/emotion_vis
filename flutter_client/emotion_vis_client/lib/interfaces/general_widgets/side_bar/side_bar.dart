import 'package:emotion_vis_client/interfaces/constants/colors.dart';
import 'package:flutter/material.dart';
import 'package:glass_kit/glass_kit.dart';

class SideBar extends StatefulWidget {
  final double width;
  final double barWidth;
  final List<ViewTab> tabs;
  final List<ActionButton> actions;
  final int selectedTab;
  const SideBar({
    Key key,
    @required this.tabs,
    this.barWidth = 60,
    this.width = 250,
    this.selectedTab,
    this.actions = const [],
  }) : super(key: key);

  @override
  _SideBarState createState() => _SideBarState();
}

class _SideBarState extends State<SideBar> {
  @override
  initState() {
    selectedTab = widget.selectedTab;
    super.initState();
  }

  @override
  void didUpdateWidget(SideBar oldWidget) {
    selectedTab = widget.selectedTab;
    super.didUpdateWidget(oldWidget);
  }

  int selectedTab = 0;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: widget.width,
      child: Row(
        children: [
          Container(
            color: pColorPrimary,
            height: double.infinity,
            width: widget.barWidth,
            child: Column(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                _tabs(),
                _actions(),
              ],
            ),
          ),
          GlassContainer.clearGlass(
            borderColor: Colors.transparent,
            width: widget.width - widget.barWidth,
            height: double.infinity,
            child: _selectedTabOptions(),
          ),
        ],
      ),
    );
  }

  Widget _actions() {
    return ListView.separated(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 20),
      shrinkWrap: true,
      itemCount: widget.actions.length,
      separatorBuilder: (context, index) => const SizedBox(height: 10),
      itemBuilder: (context, index) {
        return IconButton(
          onPressed: widget.actions[index].onTap,
          icon: Icon(
            widget.actions[index].icon,
            size: 22,
            color: pColorLight,
          ),
        );
      },
    );
  }

  Widget _selectedTabOptions() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 10),
      child: Column(
        children: [
          Container(
            width: double.infinity,
            height: 35,
            alignment: Alignment.center,
            child: Text(
              widget.tabs[selectedTab].text,
              style: const TextStyle(
                fontSize: 17,
                fontWeight: FontWeight.w700,
                color: pColorDark,
              ),
            ),
          ),
          const SizedBox(height: 10),
          Expanded(
            child: widget.tabs[selectedTab].options != null
                ? ListView.separated(
                    shrinkWrap: true,
                    itemCount: widget.tabs[selectedTab].options.length,
                    separatorBuilder: (context, index) =>
                        const SizedBox(height: 10),
                    itemBuilder: (context, index) => Container(
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.25),
                      ),
                      child: widget.tabs[selectedTab].options[index],
                    ),
                  )
                : const SizedBox(),
          ),
        ],
      ),
    );
  }

  Widget _tabs() {
    return ListView.separated(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 20),
      shrinkWrap: true,
      itemCount: widget.tabs.length,
      separatorBuilder: (context, index) => const SizedBox(height: 10),
      itemBuilder: (context, index) {
        return GestureDetector(
          onTap: () {
            widget.tabs[index].onTap();
            setState(() {
              selectedTab = index;
            });
          },
          child: Tooltip(
            message: widget.tabs[index].text,
            child: Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                color: selectedTab == index
                    ? pColorLight.withOpacity(1)
                    : Colors.white.withOpacity(0.1),
                borderRadius: BorderRadius.circular(8),
              ),
              padding: const EdgeInsets.symmetric(
                horizontal: 8,
              ),
              child: Icon(
                widget.tabs[index].icon,
                size: 22,
                color: selectedTab == index ? pColorAccent : pColorLight,
              ),
            ),
          ),
        );
      },
    );
  }
}

class ViewTab {
  IconData icon;
  Function onTap;
  String text;
  List<Widget> options;
  ViewTab({
    @required this.icon,
    @required this.onTap,
    @required this.text,
    this.options,
  });
}

class ActionButton {
  IconData icon;
  Function onTap;
  ActionButton({
    @required this.icon,
    @required this.onTap,
  });
}
