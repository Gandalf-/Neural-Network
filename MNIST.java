import java.io.IOException;
import org.apache.commons.math3.linear.*;

public class MNIST {

  private static int g_train_size    = 60000;
  private static int g_test_size     = 10000;
  private static final int image_dimension = 28 * 28;

  private static String g_train_fn = "";
  private static String g_test_fn  = "";

  /**
   * contruct and train a network for the MNIST handwriting data in csv format.
   *
   * @param  args        command line arguments
   * @throws IOException may fail if input files not found
   */
  public static void main(String[] args) throws IOException {

    final double learnSpeed = 0.50;
    final int[] description = {image_dimension, 15, 10};

    try {
      g_train_fn   = args[0];
      g_train_size = Integer.parseInt(args[1]);
      g_test_fn    = args[2];
      g_test_size  = Integer.parseInt(args[3]);

    } catch (ArrayIndexOutOfBoundsException except) {
      System.out.println("usage: java ANN mnist_train.csv mnist_test.csv");
      System.exit(1);
    }

    /* get data */
    System.out.print("Data loading...");
    final Data trainingData = new Data(g_train_fn, g_train_size, 10);
    System.out.println(" Done");

    /* build network */
    System.out.println("Learning speed: " + learnSpeed);
    final NeuralNetwork network = new NeuralNetwork(description);

    network.train(trainingData, 15, 250, learnSpeed);
    network.print_csv("models/mnist.csv");

    System.out.print("Last iteration: ");
    final Data testData     = new Data(g_test_fn,  g_test_size, 10);
    network.get_accuracy(testData);
  }
}
