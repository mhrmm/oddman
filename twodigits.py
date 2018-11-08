# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from puzzle import TwoDigitPuzzleGenerator, makePuzzleVector, InvertingTwoDigitPuzzleGenerator
from puzzle import makePuzzleTarget, buildVocab

class TwoLayerClassifier(nn.Module):  # inheriting from nn.Module!

    def __init__(self, num_labels, input_size, hidden_size):
        super(TwoLayerClassifier, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_labels)

    def forward(self, input_vec):
        nextout = self.linear1(input_vec).clamp(min=0)
        nextout = self.linear2(nextout)
        return F.log_softmax(nextout, dim=1)
    
class Trainer:
    
    def __init__(self, generator):
        self.num_training_epochs = 20
        self.training_data_size = 3000
        self.test_data_size = 100
        self.hidden_layer_size = 100
        self.num_choices = 3
        self.generator = generator

    def generateData(self):
        self.data = self.generator.getTrainingData(self.training_data_size)
        self.test_data = self.generator.getTrainingData(self.test_data_size)
        self.vocab = buildVocab(self.data + self.test_data)
        

    
    def train(self):
        self.generateData()
        model = TwoLayerClassifier(self.num_choices, 
                                   self.num_choices * len(self.vocab), 
                                   self.hidden_layer_size)
        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        for epoch in range(self.num_training_epochs):
            print('epoch {}'.format(epoch))
            for instance, label in self.data:
                # Step 1. Remember that PyTorch accumulates gradients.
                # We need to clear them out before each instance
                model.zero_grad()
        
                # Step 2. Make our input vector and also we must wrap the target in a
                # Tensor as an integer.
                input_vec = makePuzzleVector((instance, label), self.vocab)
                target = makePuzzleTarget(label)
        
                # Step 3. Run our forward pass.
                log_probs = model(input_vec)
        
                # Step 4. Compute the loss, gradients, and update the parameters by
                # calling optimizer.step()
                loss = loss_function(log_probs, target)
                loss.backward()
                optimizer.step()
        return model


    def evaluate(self, model, test_d):
        """Evaluates the trained network on test data."""
        word_to_ix = self.vocab
        with torch.no_grad():
            correct = 0
            for instance, label in test_d:
                input_vec = makePuzzleVector((instance, label), word_to_ix)
                log_probs = model(input_vec)
                probs = [math.exp(log_prob) for log_prob in log_probs.tolist()[0]]
                ranked_probs = list(zip(probs, range(len(probs))))
                response = max(ranked_probs)[1]
                if response == label:
                    correct += 1            
        return correct/len(test_d)




trainer = Trainer(InvertingTwoDigitPuzzleGenerator([0,1,2,3,4,5,6]))
model = trainer.train()    
print('training accuracy = {}'.format(trainer.evaluate(model, trainer.data)))
print('test accuracy = {}'.format(trainer.evaluate(model, trainer.test_data)))
