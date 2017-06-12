import org.apache.commons.math3.linear.*;

/**
 * encapsulates a neural network, which is composed of a number of layers;
 * each layer is composed of a number of neurons
 */
public class NeuralNetwork {

  /**
   * encapsulates information that makes up each layer of the neural network;
   * neurons are implicitly contained in the RealMatrix and RealVector
   * variables
   */
  private static class Layer {

    public int num_inputs, num_neurons;
    public RealMatrix weights;
    public RealVector outputs, biases, deltas;

    /**
     * initializes a layer by initializing each of the vectors that represent
     * the neurons in the layer and other clerical information about the
     * neurons which we need for training and predictions
     *
     * @param num_inputs number of inputs to this layer, which is the same as
     *                   the number of outputs of the previous layer
     * @param num_nodes  number of neurons for the layer
     */
    public Layer(int num_inputs, int num_nodes) {

      double[] biases_base = new double[num_nodes];
      double[][] weights_base = new double[num_nodes][num_inputs];

      for (int j = 0; j < num_nodes; j++) {
        for (int i = 0; i < num_inputs; i++)
          weights_base[j][i] = Math.random() * 0.01;
        biases_base[j] = Math.random() * 0.01;
      }

      this.num_inputs  = num_inputs;
      this.num_neurons = num_nodes;

      this.outputs = new ArrayRealVector(new double[num_nodes]);
      this.deltas  = new ArrayRealVector(new double[num_nodes]);
      this.biases  = new ArrayRealVector(biases_base);
      this.weights = new Array2DRowRealMatrix(weights_base);
    }
  }

  private static Layer[] layers;
  private static int num_layers;

  /**
   * initializes a neural network by constructing each layer given a network
   * layout description
   *
   * @param net_descr array which describes the layout of the network to
   *                  contruct. for each i greater than 0, the value at i
   *                  is the number of neurons for that layer, and the value
   *                  at i - 1 is the number of inputs to each of those
   *                  neurons
   */
  public NeuralNetwork(int[] net_descr) {

    NeuralNetwork.num_layers = net_descr.length -1;
    NeuralNetwork.layers = new Layer[NeuralNetwork.num_layers];

    for (int i = 1; i < net_descr.length; i++)
      NeuralNetwork.layers[i -1] = new Layer(net_descr[i -1], net_descr[i]);
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

    for (Layer layer : NeuralNetwork.layers)
      input = layer.outputs = sigmoid(
          layer.weights.operate(input).add(layer.biases));

    return input;
  }

  /**
   * calculates error given expected output, and propagates it through the
   * network backwards, layer by layer
   *
   * error       = expected - output
   * derivatives = output * (1 - output)
   * deltas      = output * derivatives
   *
   * @param expected   vector of expected output features
   * @see   sideEffect layer.deltas
   */
  private void back_propagate_error(RealVector expected) {

    RealVector derivatives, errors;
    int i, output_layer = NeuralNetwork.num_layers -1;

    /* for each layer in reverse */
    for (i = NeuralNetwork.num_layers -1; i >= 0; i--) {
      Layer layer = NeuralNetwork.layers[i];

      if (i == output_layer) {
        errors = expected.subtract(layer.outputs);

        derivatives = layer.outputs.ebeMultiply(
            layer.outputs.mapSubtract(1.0));

        layer.deltas = errors.ebeMultiply(derivatives).mapMultiply(-1.0);
      }
      else {
        errors = NeuralNetwork.layers[i + 1].weights.transpose().operate(
            NeuralNetwork.layers[i + 1].deltas);

        derivatives = layer.outputs.ebeMultiply(
            layer.outputs.mapSubtract(1.0));

        layer.deltas = errors.ebeMultiply(derivatives).mapMultiply(-1.0);
      }
    }
  }

  /**
   * update the network weights on a single input; deltas have already been
   * calculated by back_propagate_error()
   *
   * bias    += learn_speed + neuron.delta
   * weights += inputs * learn_speed * neuron.delta
   *
   * @param features    vector of input features
   * @param learn_speed speed at which to move in the direction of gradient
   * @see   sideEffect  layer.biases
   * @see   sideEffect  layer.weights
   */
  private void update_weights(RealVector features, double learn_speed) {

    RealVector inputs;
    int i, output_layer = 0;

    for (i = 0; i < NeuralNetwork.num_layers; i++) {
      Layer layer = NeuralNetwork.layers[i];

      if (i == output_layer)
        inputs = features;
      else
        inputs = NeuralNetwork.layers[i - 1].outputs;

      layer.biases = layer.biases.add(
          layer.deltas.mapMultiply(learn_speed));

      RealVector adjusted_deltas =
        layer.deltas.mapMultiply(learn_speed);

      RealMatrix weight_changes =
        inputs.outerProduct(adjusted_deltas).transpose();

      layer.weights = layer.weights.add(weight_changes);
    }
  }

  /**
   * runs the training loop to condition the network
   *
   * for each input:
   *   forward_propagate(input)
   *   back_propagate_error()
   *   update_weights()
   *
   * @param toler       number of iterations without improvement allowed
   * @param max_epochs  maximum iterations cap
   * @param learn_speed how quickly to move in the direction of gradient
   */
  public void train(Data data, int toler, int max_epochs, double learn_speed) {

    int i, j, dimension, epochs = 0, epochs_without_improvement = 0;

    double saved_best = Double.POSITIVE_INFINITY;
    double best_error = Double.POSITIVE_INFINITY;
    double accuracy, sum_error, best_accuracy = 0.0;

    RealVector outputs, expected, errors;

    /* precompute one hot encoding */
    double[][] one_hot_base = new double[Data.size][Data.outputs];
    for (i = 0; i < Data.size; i++)
      one_hot_base[i][(int)Data.labels[i]] = 1.0;
    RealMatrix one_hot = new Array2DRowRealMatrix(one_hot_base);

    /* train the network */
    while (epochs < max_epochs && epochs_without_improvement < toler) {
      sum_error = 0.0;

      for (i = 0; i < Data.size - 1; i++) {
        outputs   = forward_propagate(data.features.getRowVector(i));
        dimension = outputs.getDimension();

        expected = one_hot.getRowVector(i);
        errors   = expected.subtract(outputs);
        errors   = errors.ebeMultiply(errors);

        for (j = 0; j < dimension; j++)
          sum_error += errors.getEntry(j);

        back_propagate_error(expected);
        update_weights(Data.features.getRowVector(i), learn_speed);
      }

      epochs++;

      /* diagnostic output */
      if (sum_error < best_error) {
        System.out.printf(
            "epoch: % 4d, error: % 10.4f + ", epochs, sum_error);

        if (sum_error < saved_best - 1) {
          accuracy = get_accuracy(data);
          saved_best = sum_error;
          if (accuracy > best_accuracy)
            best_accuracy = accuracy;
        }
        else {
          System.out.println("");
        }

        best_error = sum_error;
        epochs_without_improvement = 0;
      }
      else {
        System.out.printf(
            "epoch: % 4d, error: % 10.4f -\n", epochs, sum_error);
        epochs_without_improvement++;
      }
    }
    System.out.printf("Best accuracy: % 10.4f\n", best_accuracy);
  }

  /**
   * gets the networks accuracy for a data set by scoring predictions made
   * through forward_propagate()
   *
   * @param data Object containing features and labels to be tested
   * @return     percentage accuracy for the data set provided
   */
  public double get_accuracy(Data data) {

    double correct = 0.0;

    for (int i = 0; i < Data.size; i++) {
      if (predict(Data.features.getRowVector(i)) == Data.labels[i])
        correct++;
    }

    System.out.printf("%.4f%% correct\n", correct / Data.size);
    return correct / Data.size;
  }

  /**
   * propagates input features through the network to get prediction
   *
   * @param features vector of input features
   * @return         predicted label
   */
  public int predict(RealVector features) {

    return forward_propagate(features).getMaxIndex();
  }

  /**
   * prints layers, and value of weights, output, and delta for each neuron
   */
  public void print_verbose() {

    System.out.println("Neural Network Layout");
    for (int i = 0; i < NeuralNetwork.layers.length; i++) {
      System.out.println("Layer: " + i);
      Layer layer = NeuralNetwork.layers[i];

      for (int j = 0; j < layer.num_neurons; j++){
        System.out.println("  Neuron: " + j);
        double[] weights = layer.weights.getRowVector(j).toArray();

        System.out.print("    Weights: ");
        for (int k = 0; k < weights.length; k++)
          System.out.printf("%.4f, ", weights[k]);
        System.out.println("");

        System.out.printf(
            "    Output : %.4f  Delta : %.4f\n",
            layer.outputs.getEntry(j), layer.deltas.getEntry(j));
      }
    }
    System.out.println("");
  }

  /**
   * prints layers, and number of weights, output, and delta for each neuron
   */
  public void print_terse() {

    System.out.println("Neural Network Layout");
    for (int i = 0; i < NeuralNetwork.layers.length; i++) {
      System.out.println("Layer: " + i);
      Layer layer = NeuralNetwork.layers[i];

      for (int j = 0; j < layer.num_neurons; j++){
        System.out.println("  Neuron: " + j);

        System.out.printf(
            "    Num Weights: %d,  Output : %.4f,  Delta : %.4f\n",
            layer.num_inputs,
            layer.outputs.getEntry(j),
            layer.deltas.getEntry(j));
      }
    }
    System.out.println("");
  }

  /**
   * applies the sigmoid function to each element of the input in place
   *
   * @param v        input vector
   * @see sideEffect v
   * @return         returns v with sigmoid applied to each element
   */
  private static RealVector sigmoid(RealVector v){

    for (int i = 0, length = v.getDimension(); i < length; i++)
      v.setEntry(i, 1.0 / (1.0 + Math.exp(- v.getEntry(i))) );
    return v;
  }
}
