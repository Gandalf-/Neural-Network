import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import org.apache.commons.math3.linear.*;

/**
 * encapsulates a neural network, which is composed of a number of layers;
 * each layer is composed of a number of neurons.
 */
public class NeuralNetwork {

  /**
   * encapsulates information that makes up each layer of the neural network;
   * neurons are implicitly contained in the RealMatrix and RealVector
   * variables.
   */
  private static class Layer {

    public int  numInputs;
    public int  numNeurons;
    public RealMatrix weights;
    public RealVector outputs;
    public RealVector biases;
    public RealVector deltas;

    /**
     * initializes a layer by initializing each of the vectors that represent
     * the neurons in the layer and other clerical information about the
     * neurons which we need for training and predictions.
     *
     * @param numInputs number of inputs to this layer, which is the same as
     *                   the number of outputs of the previous layer
     * @param numNodes  number of neurons for the layer
     */
    public Layer(int numInputs, int numNodes) {

      double[]   biasesBase  = new double[numNodes];
      double[][] weightsBase = new double[numNodes][numInputs];

      for (int j = 0; j < numNodes; j++) {
        for (int i = 0; i < numInputs; i++) {
          weightsBase[j][i] = Math.random() * 0.01;
        }
        biasesBase[j] = Math.random() * 0.01;
      }

      this.numInputs  = numInputs;
      this.numNeurons = numNodes;

      this.outputs = new ArrayRealVector(new double[numNodes]);
      this.deltas  = new ArrayRealVector(new double[numNodes]);
      this.biases  = new ArrayRealVector(biasesBase);
      this.weights = new Array2DRowRealMatrix(weightsBase);
    }
  }

  private static int     num_layers;
  private static Layer[] layers;

  /**
   * initializes a neural network by constructing each layer given a network
   * layout description
   *
   * @param netDescr array which describes the layout of the network to
   *                  contruct. for each i greater than 0, the value at i
   *                  is the number of neurons for that layer, and the value
   *                  at i - 1 is the number of inputs to each of those
   *                  neurons
   */
  public NeuralNetwork(int[] netDescr) {

    NeuralNetwork.num_layers = netDescr.length - 1;
    NeuralNetwork.layers     = new Layer[NeuralNetwork.num_layers];

    for (int i = 1; i < netDescr.length; i++) {
      NeuralNetwork.layers[i - 1] = new Layer(netDescr[i - 1], netDescr[i]);
    }
  }

  /**
   * reconstructs a saved neural network from a CSV file, the format used is
   * given in NeuralNetwork.print_csv()
   *
   * @param filename name of input CSV network model file
   */
  public NeuralNetwork(String filename) {

    BufferedReader reader = null;

    // get feature data from file
    try {
      reader = new BufferedReader(new FileReader(filename));

      // get the network description
      String   line      = reader.readLine();
      String[] elements  = line.split(",");
      int[]    netDescr = new int[elements.length];

      for (int i = 0; i < elements.length; i++) {
        netDescr[i] = Integer.parseInt(elements[i].trim());
      }

      // construct skeleton network
      NeuralNetwork.num_layers = netDescr.length - 1;
      NeuralNetwork.layers     = new Layer[NeuralNetwork.num_layers];

      for (int i = 1; i < netDescr.length; i++) {
        NeuralNetwork.layers[i - 1] = new Layer(netDescr[i - 1], netDescr[i]);
      }

      // set network values with values from file
      while ((line = reader.readLine()) != null) {
        elements = line.split(",");

        int layerIx  = Integer.parseInt(elements[0].trim());
        int neuronIx = Integer.parseInt(elements[1].trim());

        Layer layer   = NeuralNetwork.layers[layerIx];
        int numInputs = layer.numInputs;

        double[] weights = new double[numInputs];
        int index = 0;
        for (; index < numInputs; index++) {
          weights[index] = Double.parseDouble(elements[index + 2].trim());
        }

        layer.weights.setRow(neuronIx, weights);

        double output = Double.parseDouble(elements[index + 2].trim());
        layer.outputs.setEntry(neuronIx, output);

        double bias   = Double.parseDouble(elements[index + 3].trim());
        layer.biases.setEntry(neuronIx, bias);

        double delta  = Double.parseDouble(elements[index + 4].trim());
        layer.deltas.setEntry(neuronIx, delta);
      }

      reader.close();

    } catch (ArrayIndexOutOfBoundsException | IOException except) {
      System.out.println(
          "error: file \"" + filename + "\" is incorrectly formatted.");
      System.exit(1);

    } finally {
      if (reader == null) {
        System.out.println("error: could not open " + filename);
        System.exit(1);
      }
    }
  }

  /**
   * propagates the input features through the network, layer by layer
   * layer_output = sigmoid((weights * input) + bias)
   *
   * @param input      vector of input features
   * @see   sideEffect layer.outputs
   * @return           vector of output resulting from the last layer
   */
  public RealVector forward_propagate(RealVector input) {

    for (Layer layer : NeuralNetwork.layers) {
      input = layer.outputs = sigmoid(
          layer.weights.operate(input).add(layer.biases));
    }

    return input;
  }

  /**
   * calculates error given expected output, and propagates it through the
   * network backwards, layer by layer.
   * error       = expected - output
   * derivatives = output * (1 - output)
   * deltas      = output * derivatives
   *
   * @param expected   vector of expected output features
   * @see   sideEffect layer.deltas
   */
  private void back_propagate_error(RealVector expected) {

    int outputLayer = NeuralNetwork.num_layers - 1;

    /* for each layer in reverse */
    for (int i = NeuralNetwork.num_layers - 1; i >= 0; i--) {
      Layer layer = NeuralNetwork.layers[i];
      RealVector errors;

      if (i == outputLayer) {
        errors = expected.subtract(layer.outputs);

      } else {
        errors = NeuralNetwork.layers[i + 1].weights.transpose().operate(
            NeuralNetwork.layers[i + 1].deltas);
      }

      RealVector derivatives = layer.outputs.ebeMultiply(
          layer.outputs.mapSubtract(1.0));

      layer.deltas = errors.ebeMultiply(derivatives).mapMultiply(-1.0);
    }
  }

  /**
   * update the network weights on a single input; deltas have already been
   * calculated by back_propagate_error().
   * bias    += learnSpeed + neuron.delta
   * weights += inputs * learnSpeed * neuron.delta
   *
   * @param features    vector of input features
   * @param learnSpeed speed at which to move in the direction of gradient
   * @see   sideEffect  layer.biases
   * @see   sideEffect  layer.weights
   */
  private void update_weights(RealVector features, double learnSpeed) {

    int outputLayer = 0;

    for (int i = 0; i < NeuralNetwork.num_layers; i++) {
      Layer layer = NeuralNetwork.layers[i];

      RealVector inputs = (i == outputLayer)
          ? features : NeuralNetwork.layers[i - 1].outputs;

      layer.biases = layer.biases.add(layer.deltas.mapMultiply(learnSpeed));

      RealVector adjustedDeltas = layer.deltas.mapMultiply(learnSpeed);
      RealMatrix weightChanges = inputs.outerProduct(adjustedDeltas).transpose();

      layer.weights = layer.weights.add(weightChanges);
    }
  }

  /**
   * runs the training loop to condition the network.
   * for each input:
   *   forward_propagate(input)
   *   back_propagate_error()
   *   update_weights()
   *
   * @param toler       number of iterations without improvement allowed
   * @param maxEpochs  maximum iterations cap
   * @param learnSpeed how quickly to move in the direction of gradient
   */
  public void train(Data data, int toler, int maxEpochs, double learnSpeed) {

    int epochs = 0;
    int epochsWithoutImprovement = 0;

    double savedBest = Double.POSITIVE_INFINITY;
    double bestError = Double.POSITIVE_INFINITY;
    double bestAccuracy = 0.0;

    RealVector outputs, expected, errors;

    /* precompute one hot encoding */
    double[][] oneHotBase = new double[Data.size][Data.outputs];
    for (int i = 0; i < Data.size; i++) {
      oneHotBase[i][(int)Data.labels[i]] = 1.0;
    }
    RealMatrix oneHot = new Array2DRowRealMatrix(oneHotBase);

    /* train the network */
    while (epochs < maxEpochs && epochsWithoutImprovement < toler) {
      double sumError = 0.0;

      for (int i = 0; i < Data.size - 1; i++) {
        outputs   = forward_propagate(data.features.getRowVector(i));
        final int dimension = outputs.getDimension();

        expected = oneHot.getRowVector(i);
        errors   = expected.subtract(outputs);
        errors   = errors.ebeMultiply(errors);

        for (int j = 0; j < dimension; j++) {
          sumError += errors.getEntry(j);
        }

        back_propagate_error(expected);
        update_weights(Data.features.getRowVector(i), learnSpeed);
      }

      epochs++;

      /* diagnostic output */
      if (sumError < bestError) {
        System.out.printf(
            "epoch: % 4d, error: % 10.4f + ", epochs, sumError);

        if (sumError < savedBest - 1) {
          double accuracy = get_accuracy(data);
          savedBest = sumError;

          bestAccuracy =
            (accuracy > bestAccuracy) ?  accuracy : bestAccuracy;
        } else {
          System.out.println("");
        }

        bestError = sumError;
        epochsWithoutImprovement = 0;
      } else {
        System.out.printf(
            "epoch: % 4d, error: % 10.4f -\n", epochs, sumError);
        epochsWithoutImprovement++;
      }
    }
    System.out.printf("Best accuracy: % 10.4f\n", bestAccuracy);
  }

  /**
   * gets the networks accuracy for a data set by scoring predictions made
   * through forward_propagate().
   *
   * @param data Object containing features and labels to be tested
   * @return     percentage accuracy for the data set provided
   */
  public double get_accuracy(Data data) {

    double correct = 0.0;

    for (int i = 0; i < Data.size; i++) {
      if (predict(Data.features.getRowVector(i)) == Data.labels[i]) {
        correct++;
      }
    }

    System.out.printf("%.4f%% correct\n", correct / Data.size);
    return correct / Data.size;
  }

  /**
   * propagates input features through the network to get prediction.
   *
   * @param features vector of input features
   * @return         predicted label
   */
  public int predict(RealVector features) {

    return forward_propagate(features).getMaxIndex();
  }

  /**
   * prints layers, and value of weights, output, and delta for each neuron.
   */
  public void print_verbose() {

    System.out.println("Neural Network Layout");
    for (int i = 0; i < NeuralNetwork.layers.length; i++) {
      System.out.println("Layer: " + i);
      Layer layer = NeuralNetwork.layers[i];

      for (int j = 0; j < layer.numNeurons; j++) {
        System.out.println("  Neuron: " + j);
        double[] weights = layer.weights.getRowVector(j).toArray();

        System.out.print("    Weights: ");
        for (int k = 0; k < weights.length; k++) {
          System.out.printf("%.4f, ", weights[k]);
        }
        System.out.println("");

        System.out.printf(
            "    Output : %.4f  Delta : %.4f\n",
            layer.outputs.getEntry(j), layer.deltas.getEntry(j));
      }
    }
    System.out.println("");
  }

  /**
   * prints network information in CSV format.
   * layer 0 inputs, layer 1 inputs, -, layer n inputs, layer n outputs
   * layer_index, neuron_index, [weights, ...], output, bias, delta
   * ...
   *
   * @param outputFile name of the output CSV file
   */
  public void print_csv(String outputFile) {

    try {
      PrintWriter writer = new PrintWriter(outputFile, "UTF-8");

      // write the network description
      // inputs, inputs, ..., outputs
      for (int i = 0; i < NeuralNetwork.layers.length; i++) {
        writer.printf("%d,", NeuralNetwork.layers[i].numInputs);

        if (i + 1 == NeuralNetwork.layers.length) {
          writer.printf("%d\n", NeuralNetwork.layers[i].numNeurons);
        }
      }

      // for each layer
      for (int i = 0; i < NeuralNetwork.layers.length; i++) {
        Layer layer = NeuralNetwork.layers[i];

        // for each neuron
        for (int j = 0; j < layer.numNeurons; j++) {
          double [] weights = layer.weights.getRow(j);

          // write layer_index, neuron_index
          writer.printf("%d,%d,", i, j);

          // write weights
          for (int k = 0; k < weights.length; k++) {
            writer.printf("%f,", weights[k]);
          }

          // write neuron output, bias, and delta
          writer.printf(
              "%f,%f,%f\n",
              layer.outputs.getEntry(j),
              layer.biases.getEntry(j),
              layer.deltas.getEntry(j));
        }
      }

      writer.close();
    } catch (IOException except) {
      System.out.println("Error writing network");
      except.printStackTrace();
    }
  }

  /**
   * prints layers, and number of weights, output, and delta for each neuron.
   */
  public void print_terse() {

    System.out.println("Neural Network Layout");
    for (int i = 0; i < NeuralNetwork.layers.length; i++) {
      System.out.println("Layer: " + i);
      Layer layer = NeuralNetwork.layers[i];

      for (int j = 0; j < layer.numNeurons; j++) {
        System.out.println("  Neuron: " + j);

        System.out.printf(
            "    Num Weights: %d,  Output : %.4f,  Delta : %.4f\n",
            layer.numInputs,
            layer.outputs.getEntry(j),
            layer.deltas.getEntry(j));
      }
    }
    System.out.println("");
  }

  /**
   * applies the sigmoid function to each element of the input in place.
   *
   * @param vector        input vector
   * @see sideEffect vector
   * @return         returns vector with sigmoid applied to each element
   */
  private static RealVector sigmoid(RealVector vector) {

    for (int i = 0, length = vector.getDimension(); i < length; i++) {
      vector.setEntry(i, 1.0 / (1.0 + Math.exp(- vector.getEntry(i))) );
    }
    return vector;
  }
}
