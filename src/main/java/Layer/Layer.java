package Layer;

import outputFunction.OutputFunction;

public class Layer {
  public Double[][] weights;
  OutputFunction outputFunction;

  private Double[] lastInput;
  private Double[] lastOutput;

  int perceptronsPreviousLayer;
  int perceptronsCurrentLayer;
  double learningRate;


  public Layer(int perceptronsPreviousLayer, int perceptronsCurrentLayer, OutputFunction outputfunction, double learningRate) {
    this.perceptronsPreviousLayer = perceptronsPreviousLayer;
    this.perceptronsCurrentLayer = perceptronsCurrentLayer;
    this.learningRate = learningRate;

    this.outputFunction = outputfunction;
    weights = new Double[perceptronsCurrentLayer][perceptronsPreviousLayer];
    for(int current = 0; current < perceptronsCurrentLayer; current++) {
      for(int previous = 0; previous < perceptronsPreviousLayer; previous++) {
        weights[current][previous] = new Double(Math.random());
      }
    }
  }

  public double calculateX(Double[] input, Double[] weights) {
    double X = 0;
    for(int i = 0; i < input.length; i++) {
      X += input[i] * weights[i];
    }
    return X;
  }

  public Double[] feedForward(Double[] input) {
    assert input.length == weights[0].length;

    Double[] Ylist = new Double[weights.length];
    for(int current = 0; current < weights.length; current++) {
      double X = calculateX(input, weights[current]);
      Ylist[current] = outputFunction.calculateY(X);
    }

    this.lastInput = input.clone();
    this.lastOutput = Ylist.clone();
    return Ylist;
  }

  public Double[] backPropagate(Double[] derivativesNextLayer) {
    // calculate derivative of all perceptrons together
    double[] derivatesPreviousPerceptrons = new double[perceptronsPreviousLayer];
    for(int current = 0; current < perceptronsCurrentLayer; current++) {
      for(int previous = 0; previous < perceptronsPreviousLayer; previous++) {
          // Update derivatives of perceptrons of previous rows
          derivatesPreviousPerceptrons[previous] += derivativesNextLayer[current] * weights[current][previous];

          weights[current][previous] -= learningRate * derivativesNextLayer[current] * lastInput[previous];
      }
    }

    Double[] toReturn = new Double[derivatesPreviousPerceptrons.length];
    for(int d = 0; d < derivatesPreviousPerceptrons.length; d++) {
      toReturn[d] = derivatesPreviousPerceptrons[d];
    }
    return toReturn;
  }
}
