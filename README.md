
# Neural Network

`Network.java` implements a supervised back propogration deep neural network.
Full example usage can be found in `MNIST.java` and `Sandbox.java`. The Apache
Math Commons 3.6.1 jar is required and can be found 
[http://commons.apache.org/proper/commons-math/index.html| here].

## Features
- Data is encapsulated in the Data class, from `Data.java`.  Compatible MNIST
  CSV data can be found
  [https://public.anardil.net/Projects/Neural-Network/data|here].

- Network descriptions are single dimensional arrays, where the value at each
  index denotes the number of nodes in that layer. For MNIST, the description
  looks like this: `int description[] = { 28 * 28, 15, 10 };`  Which describes
  a network with 28 * 28 nodes in the input layer, one hidden layer with 15
  nodes, and an output layer with 10 nodes. One output node for each possible
  prediction, the numbers 0-9.

- With a network description, training is easy
```
  NeuralNetwork network = new NeuralNetwork(description);

  //network.print_verbose();
  network.train(training_data, 15, 250, learn_speed);

  //network.print_verbose();
  System.out.print("Last iteration: ");
  network.get_accuracy(test_data);
```

![Alt text](data/sandbox_example.png)

