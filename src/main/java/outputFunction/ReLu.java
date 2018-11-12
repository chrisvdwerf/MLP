package outputFunction;

public class ReLu implements OutputFunction {

  public ReLu(){}

  public double calculateY(double X) {
    return X;
  }

  public double calculateDY(double Y) {
    return 1;
  }
}
