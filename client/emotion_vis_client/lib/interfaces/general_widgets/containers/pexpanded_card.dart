import 'package:emotion_vis_client/interfaces/constants/colors.dart';
import 'package:emotion_vis_client/interfaces/general_widgets/containers/pcard.dart';
import 'package:expandable/expandable.dart';
import 'package:flutter/material.dart';
import 'package:flutter_icons/flutter_icons.dart';

class PExpandedCard extends StatefulWidget {
  final Widget header;
  final Widget content;
  final Function onRedirect;
  final double borderRadius;
  const PExpandedCard({
    Key key,
    @required this.header,
    @required this.content,
    this.borderRadius = 12,
    this.onRedirect,
  }) : super(key: key);

  @override
  _PExpandedCardState createState() => _PExpandedCardState();
}

class _PExpandedCardState extends State<PExpandedCard> {
  ExpandableController controller = ExpandableController();

  @override
  Widget build(BuildContext context) {
    return PCard(
      borderRadius: widget.borderRadius,
      child: ExpandableNotifier(
        child: Expandable(
          controller: controller,
          expanded: Column(
            children: [
              _collapsed(),
              controller.expanded
                  ? widget.content
                  : CircularProgressIndicator(),
            ],
          ),
          collapsed: _collapsed(),
        ),
      ),
    );
  }

  Widget _collapsed() {
    return GestureDetector(
      behavior: HitTestBehavior.opaque,
      onTap: () {
        setState(() {
          controller.expanded = !controller.expanded;
        });
      },
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Expanded(
            child: widget.header,
          ),
          Row(
            children: [
              Container(
                width: 35,
                height: 35,
                child: TextButton(
                  style: ButtonStyle(
                    shape: MaterialStateProperty.all<RoundedRectangleBorder>(
                      RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(25.0),
                        side: BorderSide(color: pColorGray),
                      ),
                    ),
                    backgroundColor:
                        MaterialStateProperty.all<Color>(Colors.white),
                    foregroundColor:
                        MaterialStateProperty.all<Color>(Colors.black),
                  ),
                  onPressed: () {
                    setState(() {
                      controller.expanded = !controller.expanded;
                    });
                  },
                  child: Icon(
                    controller.expanded ? Feather.arrow_up : Feather.arrow_down,
                    color: pColorIcons,
                    size: 16,
                  ),
                ),
              ),
              const SizedBox(width: 8),
              Visibility(
                visible: widget.onRedirect != null,
                child: SizedBox(
                  width: 80,
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Container(
                        width: 35,
                        height: 35,
                        child: TextButton(
                          style: ButtonStyle(
                            shape: MaterialStateProperty.all<
                                RoundedRectangleBorder>(
                              RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(25.0),
                                side: BorderSide(color: pColorGray),
                              ),
                            ),
                            backgroundColor:
                                MaterialStateProperty.all<Color>(Colors.white),
                            foregroundColor:
                                MaterialStateProperty.all<Color>(Colors.black),
                          ),
                          onPressed: () {
                            widget.onRedirect();
                          },
                          child: Icon(
                            Feather.maximize_2,
                            color: pColorIcons,
                            size: 16,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
