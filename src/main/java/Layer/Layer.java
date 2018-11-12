package Layer;

import outputFunction.OutputFunction;

public class Layer {
  public double[][] weights;
  OutputFunction outputFunction;

  private double[] lastInput;
  private double[] lastOutput;

  int perceptronsPreviousLayer;
  int perceptronsCurrentLayer;
  double learningRate;


  public Layer(int perceptronsPreviousLayer, int perceptronsCurrentLayer, OutputFunction outputfunction, double learningRate) {
    this.perceptronsPreviousLayer = perceptronsPreviousLayer;
    this.perceptronsCurrentLayer = perceptronsCurrentLayer;
    this.learningRate = learningRate;

    this.outputFunction = outputfunction;
    weights = new double[perceptronsCurrentLayer][perceptronsPreviousLayer];
    for(int current = 0; current < perceptronsCurrentLayer; current++) {
      for(int previous = 0; previous < perceptronsPreviousLayer; previous++) {
        weights[current][previous] = Math.random();
      }
    }
  }

  public double calculateX(double[] input, double[] weights) {
    double X = 0;
    for(int i = 0; i < input.length; i++) {
      X += input[i] * weights[i];
    }
    return X;
  }

  public double[] feedForward(double[] input) {
    assert input.length == weights[0].length;

    double[] Ylist = new double[weights.length];
    for(int current = 0; current < weights.length; current++) {
      double X = calculateX(input, weights[current]);
      Ylist[current] = outputFunction.calculateY(X);
    }

    this.lastInput = input.clone();
    this.lastOutput = Ylist.clone();
    return Ylist;
  }

  public double[] backPropagate(double[] derivativesNextLayer) {
    // calculate derivative of all perceptrons together
    double[] derivatesPreviousPerceptrons = new double[perceptronsPreviousLayer];
    for(int current = 0; current < perceptronsCurrentLayer; current++) {
      for(int previous = 0; previous < perceptronsPreviousLayer; previous++) {
          // Update derivatives of perceptrons of previous rows
          derivatesPreviousPerceptrons[previous] += derivativesNextLayer[current] * weights[current][previous];

          weights[current][previous] -= learningRate * derivativesNextLayer[current] * lastInput[previous];
      }
    }

    double[] toReturn = new double[derivatesPreviousPerceptrons.length];
    for(int d = 0; d < derivatesPreviousPerceptrons.length; d++) {
      toReturn[d] = derivatesPreviousPerceptrons[d];
    }
    return toReturn;
  }
}
