import layers
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

        if parameters == None:
            hidden_units = [20, 20, 20, 20]
            hidden_layers = 4
        else:
            hidden_units = parameters["hidden_units"]
            hidden_layers = parameters["hidden_layers"]

        # TODO: construct your network here

        network = Network(data_layer, hidden_units, hidden_layers)

        return network

    def setup(self, training_data, parameters=None):

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
            0.1, self.network.get_modules_with_parameters())

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

        data_size = 60000
        train_losses = []

        mnist = np.load("mnist.pkl", allow_pickle=True)
        training_data = mnist["training_images"], mnist["training_labels"]

        # define batch size
        batch_size = 128

        # TODO: train the network for num_iter iterations. You should append the loss of each iteration to train_losses.
        for i in range(num_iter):

            print("Iter {}".format(i))

            index = np.random.choice(
                len(training_data[0]), batch_size, replace=False)
            train_data_perm = training_data[0][index], training_data[1][index]

            # initialize the network
            if iter == 0:
                _, _, _, _ = trainer.setup(train_data_perm)

            # get the data
            x = training_data[0][index]
            y = training_data[1][index]

            self.data_layer.set_data(x)
            self.loss_layer.set_data(y)

            append_loss = self.train_step()

            train_losses = np.append(train_losses, append_loss)

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
            20, 20, 20, 20], "hidden_layers": 4}

        mnist = np.load("mnist.pkl", allow_pickle=True)
        x_train = mnist["training_images"]
        y_train = mnist["training_labels"]
        x_test = mnist["test_images"]
        y_test = mnist["test_labels"]
        train = (x_train, y_train)
        test = (x_test, y_test)

        batch_size = 64
        training = 1

        if training == 1:
            trainer.setup(train, parameters)
            trainer.train(20000)
            np.savez("mnist_weight.npz", weight=trainer.network.state_dict())

        if training == 0:
            test_data = {'trainer': trainer}
            test_answers = {'acc_final_thresh': 0.9,
                            'num_layers': [7, 8, 9, 10, 11, 12, 13, 14, 15]}

            num_correct, num_total = public_tests.test_mnist_acc(
                "sol_mnist", test_data, test_answers)

    else:
        # DO NOT CHANGE THIS BRANCH! This branch is used for autograder.
        out = {
            'trainer': trainer
        }
        return out


if __name__ == "__main__":
    main()
    pass
