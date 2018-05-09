import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import org.apache.commons.math3.linear.*;

/**
 * encapsulates information coming out of csv's; data is also standaridzed
 * during input collection.
 */
public class Data {

  public static RealMatrix features;
  public static double[] labels;
  public static int size;
  public static int outputs;

  /**
   * bundles inputs into a class for easy access by NeuralNetwork methods.
   *
   * @param  features   matrix of input features
   * @param  labels     array of correct labels
   */
  public Data(double[][] features, double[] labels, int outputs) {

    Data.features = new Array2DRowRealMatrix(features);
    Data.labels   = labels;
    Data.size     = labels.length;
    Data.outputs  = outputs;
  }

  /**
   * bundles inputs into a class for easy access by NeuralNetwork methods.
   *
   * @param  filename   path to csv
   * @param  size       number of inputs
   * @param  outputs    number of outputs
   */
  public Data(String filename, int size, int outputs) throws IOException {

    int    imageDimension = 28 * 28;
    double[]   labels   = new double[size];
    double[][] features = new double[size][imageDimension];

    BufferedReader reader = null;

    // get feature data from file
    try {
      reader = new BufferedReader(new FileReader(filename));

      // attempt to read the required number of lines
      for (int imageCount = 0; imageCount < size; imageCount++) {
        String line = reader.readLine();

        if (line == null) {
          System.out.println("error: file \"" + filename
              + "\" does not have specified number of data points");
          System.exit(1);
        }

        // get data, split into labels and pixels
        String[] pixels = line.split(",");
        labels[imageCount] = Double.parseDouble(pixels[0]);

        double maxPixelValue = 255;
        for (int pixelCount = 1; pixelCount < imageDimension; pixelCount++) {
          features[imageCount][pixelCount] =
            Double.parseDouble(pixels[pixelCount]) / maxPixelValue;
        }
      }

    } catch (ArrayIndexOutOfBoundsException except) {
      System.out.println(
          "error: file \"" + filename
          + "\" does not have specified dimensionality");
      System.exit(1);

    } finally {
      if (reader == null) {
        System.out.println("error: could not open " + filename);
        System.exit(1);
      }
      reader.close();
    }

    Data.features = new Array2DRowRealMatrix(features);
    Data.labels   = labels;
    Data.size     = size;
    Data.outputs  = outputs;
  }
}
