import 'TSerie.dart';

class MTSerie {
  List<DateTime> dateTimes;
  Map<String, TSerie> timeSeries;

  MTSerie({this.timeSeries, this.dateTimes});
  MTSerie.fromMap(Map<String, List<double>> map, {this.dateTimes}) {
    timeSeries = {};
    List<String> varNames = map.keys.toList();
    for (var i = 0; i < varNames.length; i++) {
      timeSeries[varNames[i]] = TSerie(values: map[varNames[i]]);
    }
  }

  int get timeLength => timeSeries[timeSeries.keys.toList()[0]].length;

  // int get variablesLength => timeSeries.length;

  double at(int position, String dimension) =>
      timeSeries[dimension].at(position);

  TSerie getSerie(String dimension) => timeSeries[dimension];

  // List<String> get variablesNames => timeSeries.keys.toList();
}
