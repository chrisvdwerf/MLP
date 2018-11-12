public class Main {
  public static void main(String[] args) {
    Network network = Network.getFromFile("data");
    for(int s = 0; s < 100000; s++) {
      if(s % 100 == 0) {
        System.out.println("==== " + s + " - " + network.feedFowardSamples() + " ====");
      }
    }
  }

}
