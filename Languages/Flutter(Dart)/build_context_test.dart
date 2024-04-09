import 'package:flutter/material.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key});

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  @override
  Widget build(BuildContext context) {
    print("Root BuildContext: ${context.hashCode}");
    return Scaffold(
        body: Center(
          child: Text("Text BuildContext: ${context.hashCode}"),   // Root BuildContext와 동일
        ),
        floatingActionButton: Builder(builder: (context) {
          print("FloatingActionButton Builder BuildContext: ${context.hashCode}");
          return FloatingActionButton(
            onPressed: () => ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                backgroundColor: Colors.blue,
                content: Builder(builder: (context) {
                  return Text('SnackBar Builder BuildContext: ${context.hashCode}');
                }),
              ),
            ),
            tooltip: 'snack_bar',
            child: const Icon(Icons.access_alarm),
          );
        })
    );
  }
}
