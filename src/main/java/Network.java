import Layer.Layer;

import java.util.List;

public class Network {
  private List<Layer> layers;
  private double learningRate;

  private List<Double[]> features;
  private List<Double[]> classes;


  public Network(List<Layer> layers, double learningRate, List<Double[]> features, List<Double[]> classes) {
    this.layers = layers;
    this.learningRate = learningRate;
    this.features = features;
    this.classes = classes;
  }

  public double feedForwardSample(Double[] inputs, Double[] desired) {
    Double[] output = layers.get(0).feedForward(inputs);
    for(int l = 1; l < layers.size(); l++) {
      output = layers.get(l).feedForward(output);
    }

    Double[] derivatives = new Double[desired.length];
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
}
