import java.io.IOException;
import org.apache.commons.math3.linear.*;

public class MNIST {

  private static int g_train_size    = 60000;
  private static int g_test_size     = 10000;
  private static int image_dimension = 28 * 28;

  private static String g_train_fn = "";
  private static String g_test_fn  = "";

  /**
   * contruct and train a network for the MNIST handwriting data in csv format
   *
   * @param  args        command line arguments
   * @throws IOException may fail if input files not found
   */
  public static void main (String[] args) throws IOException {

    Data training_data, test_data;
    double learn_speed = 0.50;
    int[] description = {image_dimension, 15, 10};

    try {
      g_train_fn   = args[0];
      g_train_size = Integer.parseInt(args[1]);
      g_test_fn    = args[2];
      g_test_size  = Integer.parseInt(args[3]);
    }
    catch (ArrayIndexOutOfBoundsException e) {
      System.out.println("usage: java ANN mnist_train.csv mnist_test.csv");
      System.exit(1);
    }

    /* get data */
    System.out.print("Data loading...");
    training_data = new Data(g_train_fn, g_train_size, 10);
    test_data     = new Data(g_test_fn,  g_test_size, 10);
    System.out.println(" Done");

    /* build network */
    System.out.println("Learning speed: " + learn_speed);
    NeuralNetwork network = new NeuralNetwork(description);

    //network.print_verbose();
    network.train(training_data, 15, 250, learn_speed);

    //network.print_verbose();
    System.out.print("Last iteration: ");
    network.get_accuracy(test_data);
  }
}
