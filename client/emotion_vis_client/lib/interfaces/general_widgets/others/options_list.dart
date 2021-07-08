import 'package:emotion_vis_client/interfaces/constants/colors.dart';
import 'package:flutter/material.dart';

typedef SelectedCallback = void Function(int index);

class OptionsList extends StatefulWidget {
  final List<String> optionsTitles;
  final SelectedCallback onSelected;
  OptionsList({Key key, this.optionsTitles, this.onSelected}) : super(key: key);

  @override
  _OptionsListState createState() => _OptionsListState();
}

class _OptionsListState extends State<OptionsList> {
  int selectedIndex = 0;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: List.generate(
        widget.optionsTitles.length,
        (index) => Container(
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
                    selectedIndex == index ? pColorPrimary : Colors.white,
                  ),
                  foregroundColor:
                      MaterialStateProperty.all<Color>(Colors.black),
                ),
                onPressed: () {
                  setState(() {
                    selectedIndex = index;
                    if (widget.onSelected != null) {
                      widget.onSelected(index);
                    }
                  });
                },
                child: Text(
                  widget.optionsTitles[index],
                  style: TextStyle(
                      color: selectedIndex == index
                          ? Color.fromARGB(255, 240, 240, 240)
                          : pTextColorPrimary),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
