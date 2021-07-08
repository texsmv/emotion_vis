import 'package:emotion_vis_client/interfaces/general_widgets/buttons/pfilled_button.dart';
import 'package:emotion_vis_client/interfaces/general_widgets/buttons/poutlined_button.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';

class PDialogProceed extends StatelessWidget {
  final Color color;
  final VoidCallback onProceed;
  final VoidCallback onCancel;
  final String title;
  final String description;
  final String footnote;
  final String proceedButtonText;
  final String cancelButtonText;

  const PDialogProceed({
    Key key,
    @required this.color,
    @required this.onProceed,
    @required this.onCancel,
    @required this.title,
    @required this.description,
    @required this.proceedButtonText,
    @required this.cancelButtonText,
    this.footnote,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 200,
      width: 320,
      padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 20),
      child: Column(
        children: [
          // const BottomSheetDragger(),
          Text(
            title,
            style: const TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.w600,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 15),
          Text(
            description,
            style: const TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w400,
            ),
            textAlign: TextAlign.justify,
          ),
          const Expanded(child: SizedBox(height: 5)),
          Row(
            children: [
              SizedBox(
                width: 100,
                height: 40,
                child: PFilledButton(
                  buttonColor: color,
                  text: cancelButtonText,
                  onPressed: onCancel,
                ),
              ),
              const SizedBox(width: 10),
              SizedBox(
                width: 100,
                height: 40,
                child: POutlinedButton(
                  buttonColor: color,
                  text: proceedButtonText,
                  onPressed: onProceed,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
