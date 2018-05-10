import java.util.NoSuchElementException;
import java.util.Scanner;
import org.apache.commons.math3.linear.*;

public class Sandbox {

  /**
   * generates a Data object for testing.
   */
  private static Data make_trainData() {
    int trSamples = 10000;
    int teSamples = 1000;

    double[][] trDataFeat  = new double[trSamples][3];
    double[]   trDataLabel = new double[trSamples];

    double[][] teDataFeat  = new double[teSamples][3];
    double[]   teDataLabel = new double[teSamples];

    /* make the training data */
    for (int i = 0; i < trSamples; i++) {
      double val = Math.random() - 0.5;
      trDataFeat[i][0] = val;
      trDataFeat[i][1] = val * val;
      trDataFeat[i][2] = val * val * val;

      if (val < 0) {
        trDataLabel[i] = 0.0;
      } else {
        trDataLabel[i] = 1.0;
      }
    }

    return new Data(trDataFeat, trDataLabel, 2);
  }

  /**
   * run some tests.
   */
  public static void main(String[] args) {

    /* get data */
    Data trainingData = make_trainData();

    /* build the network */
    final int[] description = {3, 3, 2, 2};
    //NeuralNetwork network = new NeuralNetwork(description);
    final NeuralNetwork network = new NeuralNetwork("models/sandbox.csv");

    /* train */
    final int tolerance  = 15;
    final int maxEpochs = 200;
    final double speed   = 0.01;
    network.train(trainingData, tolerance, maxEpochs, speed);

    /* final output */
    System.out.print("Last iteration: ");
    network.get_accuracy(trainingData);
    network.print_csv("models/sandbox.csv");

    /* interactive interface with resulting model */
    System.out.println("Starting interactive mode");
    final Scanner scanner = new Scanner(System.in);

    while (true) {
      try {
        final double   val   = scanner.nextDouble();
        final double[] feats = {val, val * val, val * val * val};
        final int prediction = network.predict(new ArrayRealVector(feats));

        if (prediction == 1) {
          System.out.println("postive!");

        } else if (prediction == 0) {
          System.out.println("negative!");
        }

      } catch (NoSuchElementException except) {
        System.out.println("error: input must be number");
        scanner.next();
      }

      System.out.println("");
    }
  }
}
