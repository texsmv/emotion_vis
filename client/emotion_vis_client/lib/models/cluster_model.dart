import 'dart:ui';

import 'package:emotion_vis_client/models/person_model.dart';

class ClusterModel {
  String id;
  Color color;
  List<PersonModel> persons;
  ClusterModel({this.id, this.color, this.persons});

  List<String> get personsIds =>
      List.generate(persons.length, (index) => persons[index].id);
}
