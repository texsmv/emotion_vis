import 'package:emotion_vis_client/interfaces/constants/colors.dart';
import 'package:flutter/material.dart';

class OptionsBar extends StatefulWidget {
  final List<OptionTab> tabs;
  final List<OptionButton> buttons;
  final List<OptionButton> trailingButtons;
  const OptionsBar({Key key, this.tabs, this.buttons, this.trailingButtons})
      : super(key: key);

  @override
  _OptionsBarState createState() => _OptionsBarState();
}

class _OptionsBarState extends State<OptionsBar> {
  int selectedTab = 0;

  @override
  void initState() {
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 160,
      height: double.infinity,
      color: pColorBackground,
      // color: Colors.red,
      child: Column(
        children: [
          widget.tabs != null
              ? ListView.separated(
                  padding: EdgeInsets.symmetric(horizontal: 10, vertical: 20),
                  shrinkWrap: true,
                  itemCount: widget.tabs.length,
                  separatorBuilder: (context, index) => SizedBox(height: 10),
                  itemBuilder: (context, index) {
                    return GestureDetector(
                      onTap: () {
                        widget.tabs[index].onTap();
                        setState(() {
                          selectedTab = index;
                        });
                      },
                      child: Container(
                        width: double.infinity,
                        height: 40,
                        decoration: BoxDecoration(
                          color: selectedTab == index
                              ? Color.fromARGB(255, 235, 235, 235)
                              : pColorBackground,
                          borderRadius: BorderRadius.circular(8),
                        ),
                        padding: EdgeInsets.symmetric(
                          horizontal: 8,
                        ),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.start,
                          children: [
                            Icon(
                              widget.tabs[index].icon,
                              size: 22,
                              color: pColorIcons,
                            ),
                            SizedBox(width: 5),
                            Text(
                              widget.tabs[index].text,
                              style: TextStyle(
                                fontSize: 14,
                                color: pTextColorSecondary,
                              ),
                            ),
                          ],
                        ),
                      ),
                    );
                  },
                )
              : SizedBox(),
          widget.buttons != null
              ? Column(
                  children: [
                    Divider(),
                    ListView.separated(
                      padding:
                          EdgeInsets.symmetric(horizontal: 10, vertical: 10),
                      shrinkWrap: true,
                      itemCount: widget.buttons.length,
                      separatorBuilder: (context, index) => SizedBox(height: 0),
                      itemBuilder: (context, index) {
                        if (widget.buttons[index].custom != null) {
                          return widget.buttons[index].custom;
                        }
                        return TextButton(
                          style: ButtonStyle(
                            backgroundColor:
                                MaterialStateProperty.all<Color>(Colors.white),
                            foregroundColor:
                                MaterialStateProperty.all<Color>(Colors.black),
                          ),
                          onPressed: widget.buttons[index].onTap,
                          child: Row(
                            crossAxisAlignment: CrossAxisAlignment.center,
                            children: [
                              Icon(
                                widget.buttons[index].icon,
                                size: 22,
                                color: pColorIcons,
                              ),
                              SizedBox(width: 8),
                              widget.buttons[index].text != null
                                  ? Column(
                                      mainAxisAlignment:
                                          MainAxisAlignment.spaceAround,
                                      children: [
                                        SizedBox(
                                          height: 10,
                                        ),
                                        Text(
                                          widget.buttons[index].text,
                                          style: TextStyle(
                                            fontSize: 13,
                                            color: pTextColorSecondary,
                                          ),
                                        ),
                                        widget.buttons[index].option != null
                                            ? Text(
                                                widget.buttons[index].option,
                                                style: TextStyle(
                                                  fontSize: 11,
                                                  color: Colors.black,
                                                ),
                                              )
                                            : SizedBox(),
                                      ],
                                    )
                                  : SizedBox(),
                            ],
                          ),
                        );
                      },
                    ),
                  ],
                )
              : SizedBox(),
          Expanded(child: SizedBox()),
          widget.trailingButtons != null
              ? Column(
                  children: [
                    Divider(),
                    ListView.separated(
                      padding:
                          EdgeInsets.symmetric(horizontal: 10, vertical: 20),
                      shrinkWrap: true,
                      itemCount: widget.trailingButtons.length,
                      separatorBuilder: (context, index) =>
                          SizedBox(height: 10),
                      itemBuilder: (context, index) {
                        return InkWell(
                          child: IconButton(
                            splashColor: Colors.grey,
                            highlightColor: Colors.grey,
                            focusColor: Colors.grey,
                            color: Colors.blue,
                            hoverColor: Colors.grey,
                            icon: Icon(
                              widget.trailingButtons[index].icon,
                              size: 28,
                              color: pColorIcons,
                            ),
                            onPressed: widget.trailingButtons[index].onTap,
                          ),
                        );
                      },
                    ),
                  ],
                )
              : SizedBox(),
        ],
      ),
    );
  }
}

class OptionTab {
  IconData icon;
  Function onTap;
  String text;
  OptionTab({@required this.icon, @required this.onTap, @required this.text});
}

class OptionButton {
  IconData icon;
  Function onTap;
  String text;
  String option;
  Widget custom;
  OptionButton(
      {@required this.icon,
      @required this.onTap,
      this.text,
      this.option,
      this.custom});
}
