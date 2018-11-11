package outputFunction;

public class ReLu implements OutputFunction {

  public ReLu(){}

  public double calculateY(double X) {
    return Math.max(0, X);
  }

  public double calculateDY(double Y) {
    return 1;
  }
}
