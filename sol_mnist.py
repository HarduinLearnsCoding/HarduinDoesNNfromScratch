import layers as layers
import numpy as np
import matplotlib.pyplot as plt
import data_generators
import public_tests
import pickle
import random


class Network(layers.BaseNetwork):
    # TODO: you might need to pass additional arguments to init for prob 2, 3, 4 and mnist

    def __init__(self, data_layer, hidden_units, hidden_layers):

        # you should always call __init__ first

        super().__init__()

        # TODO: define our network architecture here

        # self.layer_2_linear = layers.Linear(data_layer, hidden_units)
        # self.layer_3_bias = layers.Bias(self.layer_2_linear)
        # self.layer_4_relu = layers.Relu(self.layer_3_bias)
        # self.layer_5_linear = layers.Linear(self.layer_4_relu, 1)
        # self.layer_6_bias = layers.Bias(self.layer_5_linear)

        self.MY_MODULE_LIST = layers.ModuleList()
        self.MY_MODULE_LIST.append(data_layer)
        for i in range(hidden_layers):
            self.MY_MODULE_LIST.append(layers.Linear(
                self.MY_MODULE_LIST[-1], hidden_units[i]))
            self.MY_MODULE_LIST.append(layers.Bias(self.MY_MODULE_LIST[-1]))
            self.MY_MODULE_LIST.append(layers.Relu(self.MY_MODULE_LIST[-1]))

        # self.layer_4_loss = layers.SquareLoss(self.layer_3_bias)
        self.MY_MODULE_LIST.append(layers.Linear(self.MY_MODULE_LIST[-1], 10))
        self.MY_MODULE_LIST.append(layers.Bias(self.MY_MODULE_LIST[-1]))

        # TODO: always call self.set_output_layer with the output layer of this network (usually the last layer)

        self.set_output_layer(self.MY_MODULE_LIST[-1])


class Trainer:

    def __init__(self):
        pass

    def define_network(self, data_layer, parameters=None):

        # For prob 2, 3, 4 and mnist:
        # parameters is a dict that might contain keys: "hidden_units" and "hidden_layers".
        #"hidden_units" specify the number of hidden units for each layer. "hidden_layers" specify number of hidden layers.
        # Note: we might be testing if your network code is generic enough through define_network. Your network code can be even more general, but this is the bare minimum you need to support.
        # Note: You are not required to use define_network in setup function below, although you are welcome to.

        # needed for prob 2, 3, 4, mnist

        if parameters == None:
            hidden_units = [120, 100, 100, 80, 80]
            hidden_layers = 5
        else:
            hidden_units = parameters["hidden_units"]
            hidden_layers = parameters["hidden_layers"]

        # TODO: construct your network here

        network = Network(data_layer, hidden_units, hidden_layers)

        return network

    def setup(self, training_data, parameters=None, batch_size=64):

        x, y = training_data

        # TODO: define input data layer

        self.data_layer = layers.Data(x)

        # TODO: construct the network. you don't have to use define_network.

        self.network = self.define_network(self.data_layer, parameters)

        # TODO: use the appropriate loss function here

        self.loss_layer = layers.CrossEntropySoftMax(
            self.network.get_output_layer())

        # TODO: construct the optimizer class here. You can retrieve all modules with parameters (thus need to be optimized be the optimizer) by "network.get_modules_with_parameters()"

        self.optim = layers.SGDSolver(
            0.4, self.network.get_modules_with_parameters())

        return self.data_layer, self.network, self.loss_layer, self.optim

    def train_step(self):
        # TODO: train the network for a single iteration
        # you have to return loss for the function
        loss = self.loss_layer.forward()
        self.loss_layer.backward()
        self.optim.step()

        return loss

    def get_num_iters_on_public_test(self):
        # TODO: adjust this number to how much iterations you want to train on the public test dataset for this problem.
        return 10

    def train(self, num_iter):

        # print(batch_size)

        # epochs = 5
        data_size = 60000
        train_losses = []
        batch_size = 100
        mnist = np.load("mnist.pkl", allow_pickle=True)
        x = mnist["training_images"]
        y = mnist["training_labels"]
        data_size = x.shape[0]

        # TODO: train the network for num_iter iterations. You should append the loss of each iteration to train_losses.
        for i in range(num_iter):

            print("Epoch {}".format(i))

            permutations = np.random.permutation(data_size)
            x_train = x[permutations]
            y_train = y[permutations]

            for j in range(0, data_size, batch_size):

                # print("Batch: ", j)
                x_batch = x_train[j:j + batch_size]

                y_batch = y_train[j:j + batch_size]

                self.data_layer.set_data(x_batch)
                self.loss_layer.set_data(y_batch)

                # print(y_train.shape)

                append_loss = self.train_step()
                train_losses = np.append(train_losses, append_loss)

        # np.savez("mnist_weight.npz", weight=trainer.network.state_dict())
        # you have to return train_losses for the function
        print("Losses", train_losses)
        return train_losses

# DO NOT CHANGE THE NAME OF THIS FUNCTION


def main(test=False):

    # setup the trainer

    trainer = Trainer()

    # DO NOT REMOVE THESE IF/ELSE
    if not test:

        # Your code goes here.

        parameters = {"hidden_units": [
            120, 100, 100, 80, 80], "hidden_layers": 5}

        mnist = np.load("mnist.pkl", allow_pickle=True)
        x_train = mnist["training_images"]
        y_train = mnist["training_labels"]
        x_test = mnist["test_images"]
        y_test = mnist["test_labels"]
        train = (x_train, y_train)
        test = (x_test, y_test)

        # print('Labels Type', type(y_train))

        batch_size = 64
        # Mini Batch here?
        training = 0

        if training == 1:
            trainer.setup(train, parameters, batch_size)
            trainer.train(10)
            np.savez("mnist_weight.npz",
                     weight=trainer.network.state_dict())

        if training == 0:
            test_data = {'trainer': trainer}
            test_answers = {'acc_final_thresh': 0.9,
                            'num_layers': [7, 8, 9, 10, 11, 12, 13, 14, 15]}

            # Network Architecture Public Test
            num_correct, num_total = public_tests.test_network_arch(
                "sol_mnist", test_data, test_answers)

            print("Architecture", num_correct, "/", num_total)

            num_correct, num_total = public_tests.test_mnist_acc(
                "sol_mnist", test_data, test_answers)
            print("Acc Threshold Pass/Fail", num_correct, "/", num_total)

        # num_correct, num_total = public_tests.test_gradients(
        #     "sol_mnist", test_data, test_answers)
        # print("Gradient Pass/Fail", num_correct, "/", num_total, "\n")

        # For prob mnist: you can use this snippet to save the network weight:
        # np.savez("mnist_weight.npz", weight=trainer.network.state_dict())
    else:
        # DO NOT CHANGE THIS BRANCH! This branch is used for autograder.
        out = {
            'trainer': trainer
        }
        return out


if __name__ == "__main__":
    main()
    pass
