import time

import torch
import torch.nn as nn
import torch.optim as optim


class NeuralNetworkTorch(nn.Module):
    def __init__(self, sizes, epochs=10, learning_rate=0.01, random_state=1):
       
        super().__init__()

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.random_state = random_state   
        torch.manual_seed(self.random_state)

        '''
        TODO: Implement the forward propagation algorithm.
        The layers should be initialized according to the sizes variable.
        The layers should be implemented using variable size analogously to
        the implementation in network_pytorch: sizes[i] is the shape 
        of the input that gets multiplied to the weights for the layer.
        '''
        self.layers = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

        self.activation_func = torch.sigmoid
        self.output_func = torch.softmax
        self.loss_func = nn.BCEWithLogitsLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)

        
    def _forward_pass(self, x_train):
        '''
        TODO: The method should return the output of the network.
        '''
        for layer in self.layers[:-1]:
            x_train = self.activation_func(layer(x_train))
        return self.layers[-1](x_train)



    def _backward_pass(self, y_train, output):
        '''
        TODO: Implement the backpropagation algorithm responsible for updating the weights of the neural network.
        '''
        loss = self.loss_func(output, y_train.float())
        loss.backward()




    def _update_weights(self):
        '''
        TODO: Update the network weights according to stochastic gradient descent.
        '''
        self.optimizer.step()


    def _flatten(self, x):
        return x.view(x.size(0), -1)       


    def _print_learning_progress(self, start_time, iteration, train_loader, val_loader):
        train_accuracy = self.compute_accuracy(train_loader)
        val_accuracy = self.compute_accuracy(val_loader)
        print(
            f'Epoch: {iteration + 1}, ' \
            f'Training Time: {time.time() - start_time:.2f}s, ' \
            f'Learning Rate: {self.optimizer.param_groups[0]["lr"]}, ' \
            f'Training Accuracy: {train_accuracy * 100:.2f}%, ' \
            f'Validation Accuracy: {val_accuracy * 100:.2f}%'
            )
        return train_accuracy, val_accuracy


    def predict(self, x):
        '''
        TODO: Implement the prediction making of the network.
        The method should return the index of the most likeliest output class.
        '''
        x = self._flatten(x)
        output = self._forward_pass(x)
        return torch.argmax(output, axis=1)



    def fit(self, train_loader, val_loader):
        start_time = time.time()
        history = {'accuracy': [], 'val_accuracy': []} 

        for iteration in range(self.epochs): 
            for x, y in train_loader:
                x = self._flatten(x)
                y = nn.functional.one_hot(y, 10)
                self.optimizer.zero_grad()


                output = self._forward_pass(x) 
                self._backward_pass(y, output)
                self._update_weights()

            train_accuracy, val_accuracy = self._print_learning_progress(start_time, iteration, train_loader, val_loader)
            history['accuracy'].append(train_accuracy)
            history['val_accuracy'].append(val_accuracy)

        return history



    def compute_accuracy(self, data_loader):
        correct = 0
        for x, y in data_loader:
            pred = self.predict(x)
            correct += torch.sum(torch.eq(pred, y))

        return correct / len(data_loader.dataset)
