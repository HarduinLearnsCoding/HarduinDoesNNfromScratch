import layers
import numpy as np
import matplotlib.pyplot as plt
import data_generators
import public_tests


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

        self.MY_MODULE_LIST.append(layers.Linear(self.MY_MODULE_LIST[-1], 1))
        self.MY_MODULE_LIST.append(layers.Bias(self.MY_MODULE_LIST[-1]))
        # TODO: always call self.set_output_layer with the output layer of this network (usually the last layer)

        self.set_output_layer(self.MY_MODULE_LIST[-1])


class Trainer:

    def __init__(self):
        pass

    def define_network(self, data_layer, parameters=None):

        if parameters == None:
            hidden_units = [10, 8, 6]
            hidden_layers = 3
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
        return 60000

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

        parameters = {"hidden_units": [10, 8, 6], "hidden_layers": 3}

        dict_train_test = data_generators.data_3b()
        train = dict_train_test["train"]
        test = dict_train_test["test"]
        x, y = train
        xtest, ytest = test

        trainer.setup(train, parameters)
        # trainer.train(90000)
        test_data = {'trainer': trainer}
        test_answers = {'loss_final_thresh': 0.001,
                        'num_layers': [7, 8, 9, 10, 11, 12, 13, 14, 15]}

        # Network Architecture Public Test
        num_correct, num_total = public_tests.test_network_arch(
            "sol_3b", test_data, test_answers)

        print("Architecture", num_correct, "/", num_total)

        num_correct, num_total = public_tests.test_final_mse(
            "sol_3b", test_data, test_answers)
        print("MSE Threshold Pass/Fail", num_correct, "/", num_total)

        num_correct, num_total = public_tests.test_gradients(
            "sol_3b", test_data, test_answers)
        print("Gradient Pass/Fail", num_correct, "/", num_total, "\n")

        # if plot == 1:
        #     test_data = {'trainer': trainer}
        #     experiments.plot_tvp("sol_3b", test_data,
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
