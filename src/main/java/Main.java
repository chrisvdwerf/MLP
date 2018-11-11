import Layer.Layer;
import outputFunction.ReLu;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Main {
  public static void main(String[] args) {
    Network network = getFromFile("data");

    for(int s = 0; s < 100000; s++) {
      if(s % 100 == 0) {
        System.out.println("==== " + s + " - " + network.feedFowardSamples() + " ====");
      }
    }
  }

  public static Network getFromFile(String filename) {
    try {
      BufferedReader reader = new BufferedReader(new FileReader(new File(Main.class.getResource(filename).getFile())));
      double learningRate = Double.parseDouble(reader.readLine());
      int inputs = Integer.valueOf(reader.readLine());

      List<String> configuration = readCSVLine(reader.readLine());

      List<Layer> layers = new ArrayList<Layer>();
      layers.add(new Layer(inputs, Integer.valueOf(configuration.get(0)), new ReLu(), learningRate));
      for(int l = 1; l < configuration.size(); l++) {
        layers.add(new Layer(Integer.valueOf(configuration.get(l - 1)), Integer.valueOf(configuration.get(l)), new ReLu(), learningRate));
      }

      List<Double[]> features = new ArrayList<Double[]>();
      List<Double[]> classes = new ArrayList<Double[]>();
      int outputs = Integer.valueOf(configuration.get(configuration.size() - 1));

      List<String> items = readCSVLine(reader.readLine());
      while(items != null && items.size() > 0) {
        Double[] featureList = new Double[inputs];
        for(int i = 0; i < inputs; i++) {
          featureList[i] = Double.valueOf(items.get(i));
        }
        features.add(featureList);

        Double[] classList = new Double[outputs];
        for(int i = 0; i < outputs; i++) {
          classList[i] = Double.valueOf(items.get(i + inputs));
        }
        classes.add(classList);
        items = readCSVLine(reader.readLine());
      }

      return new Network(layers, learningRate, features, classes);
    } catch(FileNotFoundException e) {
      e.printStackTrace();
      throw new IllegalArgumentException("File not found!", e);
    } catch(IOException e) {
      e.printStackTrace();
      throw new IllegalArgumentException("Error whilst parsing!", e);
    }
  }

  public static List<String> readCSVLine(String s) {
    System.out.println("string: " + s);
    return s != null ? Arrays.asList(s.split("\\s*,\\s*")) : null;
  }
}
