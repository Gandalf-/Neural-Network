import java.util.InputMismatchException;
import java.util.Scanner;
import org.apache.commons.math3.linear.*;

public class Sandbox {

  /**
   * generates a Data object for testing
   */
  private static Data make_train_data() {
    int tr_samples = 10000, te_samples = 1000;

    double[][] tr_data_feat  = new double[tr_samples][3];
    double[]   tr_data_label = new double[tr_samples];

    double[][] te_data_feat  = new double[te_samples][3];
    double[]   te_data_label = new double[te_samples];

    /* make the training data */
    for (int i = 0; i < tr_samples; i++) {
      double x = Math.random() - 0.5;
      tr_data_feat[i][0] = x;
      tr_data_feat[i][1] = x * x;
      tr_data_feat[i][2] = x * x * x;

      if (x < 0)
        tr_data_label[i] = 0.0;
      else
        tr_data_label[i] = 1.0;
    }

    return new Data(tr_data_feat, tr_data_label, 2);
  }

  /**
   * run some tests
   */
  public static void main(String[] args) {

    /* get data */
    Data training_data = make_train_data();

    /* build the network */
    int[] description = {3, 3, 2, 2};
    //NeuralNetwork network = new NeuralNetwork(description);
    NeuralNetwork network = new NeuralNetwork("models/sandbox.csv");

    /* train */
    int tolerance  = 15;
    int max_epochs = 200;
    double speed   = 0.01;
    network.train(training_data, tolerance, max_epochs, speed);

    /* final output */
    System.out.print("Last iteration: ");
    network.get_accuracy(training_data);
    network.print_csv("models/sandbox.csv");

    /* interactive interface with resulting model */
    System.out.println("Starting interactive mode");
    Scanner s = new Scanner(System.in);
    int prediction;

    while (true) {
      try {
        double   x     = s.nextDouble();
        double[] feats = {x, x*x, x*x*x};
        prediction     = network.predict(new ArrayRealVector(feats));

        if (prediction == 1)
          System.out.println("postive!");

        else if (prediction == 0)
          System.out.println("negative!");

      }
      catch (InputMismatchException e) {
        System.out.println("error: input must be number");
        s.next();
      }

      System.out.println("");
    }
  }
}
