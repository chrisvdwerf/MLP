import Layer.Layer;
import outputFunction.ReLu;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Network {
  private List<Layer> layers;
  private double learningRate;

  private List<double[]> features;
  private List<double[]> classes;


  public Network(List<Layer> layers, double learningRate, List<double[]> features, List<double[]> classes) {
    this.layers = layers;
    this.learningRate = learningRate;
    this.features = features;
    this.classes = classes;
  }

  public double feedForwardSample(double[] inputs, double[] desired) {
    double[] output = layers.get(0).feedForward(inputs);
    for(int l = 1; l < layers.size(); l++) {
      output = layers.get(l).feedForward(output);
    }

    double[] derivatives = new double[desired.length];
    for(int d = 0; d < derivatives.length; d++) {
      derivatives[d] = 2 * (output[d] - desired[d]);
    }

    for(int l = layers.size() - 1; l >= 0; l--) {
      derivatives = layers.get(l).backPropagate(derivatives);
    }

    double MSE = 0;
    for(int d = 0; d < desired.length; d++) {
      double out = output[d];
      double des = desired[d];
      MSE += Math.pow(out - des,2) / (1.d * desired.length);
    }

    return MSE;
  }

  public double feedFowardSamples() {
    double d = 0;
    for(int s = 0; s < features.size(); s++) {
      d += feedForwardSample(features.get(s), classes.get(s));
    }
    return d / features.size();
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

      List<double[]> features = new ArrayList<double[]>();
      List<double[]> classes = new ArrayList<double[]>();
      int outputs = Integer.valueOf(configuration.get(configuration.size() - 1));

      List<String> items = readCSVLine(reader.readLine());
      while(items != null && items.size() > 0) {
        double[] featureList = new double[inputs];
        for(int i = 0; i < inputs; i++) {
          featureList[i] = Double.valueOf(items.get(i));
        }
        features.add(featureList);

        double[] classList = new double[outputs];
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

  private static List<String> readCSVLine(String s) {
    System.out.println("string: " + s);
    return s != null ? Arrays.asList(s.split("\\s*,\\s*")) : null;
  }
}
