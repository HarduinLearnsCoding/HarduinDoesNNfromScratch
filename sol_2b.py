import layers
import numpy as np
import matplotlib.pyplot as plt
import data_generators
import public_tests


class Network(layers.BaseNetwork):
    # TODO: you might need to pass additional arguments to init for prob 2, 3, 4 and mnist

    def __init__(self, data_layer, hidden_units):

        # you should always call __init__ first

        super().__init__()

        # TODO: define our network architecture here

        self.layer_2_linear = layers.Linear(data_layer, hidden_units)
        self.layer_3_bias = layers.Bias(self.layer_2_linear)
        self.layer_4_relu = layers.Relu(self.layer_3_bias)
        self.layer_5_linear = layers.Linear(self.layer_4_relu, 1)
        self.layer_6_bias = layers.Bias(self.layer_5_linear)

        # TODO: always call self.set_output_layer with the output layer of this network (usually the last layer)

        self.set_output_layer(self.layer_6_bias)


class Trainer:

    def __init__(self):
        pass

    def define_network(self, data_layer, parameters=None):

        if parameters == None:
            hidden_units = 15
        else:
            hidden_units = parameters["hidden_units"]

        # TODO: construct your network here

        network = Network(data_layer, hidden_units)

        return network

    def setup(self, training_data, parameters=None):

        x, y = training_data

        # TODO: define input data layer

        self.data_layer = layers.Data(x)

        # TODO: construct the network. you don't have to use define_network.

        self.network = self.define_network(self.data_layer, parameters)

        # TODO: use the appropriate loss function here

        self.loss_layer = layers.SquareLoss(self.network.get_output_layer(), y)

        # TODO: construct the optimizer class here. You can retrieve all modules with parameters (thus need to be optimized be the optimizer) by "network.get_modules_with_parameters()"

        self.optim = layers.SGDSolver(
            0.5, self.network.get_modules_with_parameters())

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
        return 10000

    def train(self, num_iter):
        train_losses = []
        # TODO: train the network for num_iter iterations. You should append the loss of each iteration to train_losses.
        for i in range(0, num_iter):
            append_loss = self.train_step()
            train_losses = np.append(train_losses, append_loss)
        # you have to return train_losses for the function
        return train_losses

# DO NOT CHANGE THE NAME OF THIS FUNCTION


def main(test=False):

    # setup the trainer

    trainer = Trainer()

    # DO NOT REMOVE THESE IF/ELSE
    if not test:

        # Your code goes here.

        parameters = {"hidden_units": 15, "hidden_layers": 1}

        dict_train_test = data_generators.data_2b()
        train = dict_train_test["train"]
        test = dict_train_test["test"]
        x, y = train
        xtest, ytest = test

        trainer.setup(train, parameters)
        trainer.train(10000)

        # if plot == 1:
        #     test_data = {'trainer': trainer}
        #     experiments.plot_tvp("sol_2b", test_data,
        #                          trainer.get_num_iters_on_public_test())

    else:
        # DO NOT CHANGE THIS BRANCH! This branch is used for autograder.
        out = {
            'trainer': trainer
        }
        return out


if __name__ == "__main__":
    main()
    pass
