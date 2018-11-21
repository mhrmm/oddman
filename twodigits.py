import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from puzzle import AltTwoDigitPuzzleGenerator


if torch.cuda.is_available():
    print("using gpu")
    cuda = torch.device('cuda:0')
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    def cudaify(model):
        model.cuda()
else:
    print("using cpu")
    cuda = torch.device('cpu')
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    def cudaify(model):
        pass



def oneHot(word, vocab):
    vec = [0]*len(vocab)
    vec[vocab[word]] = 1
    return vec

def makePuzzleVector(puzzle, vocab):
    choices, _ = puzzle
    oneHotVec = []
    for choice in choices:
        oneHotVec += oneHot(str(choice), vocab)
    return FloatTensor(oneHotVec, device=cuda).view(1, -1)

def makePuzzleMatrix(puzzles, vocab):
    matrix = []
    for puzzle in puzzles:
        choices, _ = puzzle
        oneHotVec = []
        for choice in choices:
            oneHotVec += oneHot(str(choice), vocab)
        matrix.append(oneHotVec)
    return FloatTensor(matrix, device=cuda)

def makePuzzleTarget(label):
    return LongTensor([label])    

def makePuzzleTargets(labels):
    return LongTensor(labels)    

def buildVocab(puzzles):
    word_to_ix = {}
    for choices, _ in puzzles:
        for word in choices:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    return word_to_ix


class TwoLayerClassifier(nn.Module): 

    def __init__(self, num_labels, input_size, hidden_size):
        super(TwoLayerClassifier, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        #self.linearh = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_labels)

    def forward(self, input_vec):
        nextout = self.linear1(input_vec).clamp(min=0)
        #nextout = self.linearh(nextout).clamp(min=0)
        nextout = self.linear2(nextout)
        return F.log_softmax(nextout, dim=1)

    
class Trainer:
    
    def __init__(self, generator, num_choices):
        self.num_training_epochs = 1000
        self.training_data_size = 100000
        self.test_data_size = 100
        self.hidden_layer_size = 100
        self.num_choices = num_choices
        self.generator = generator

    def generateData(self):
        self.data = set(self.generator.getTrainingData(self.training_data_size))
        self.test_data = set()
        while len(self.test_data) < self.test_data_size:
            puzzle = self.generator.generate()
            if puzzle not in self.data:
                self.test_data.add(puzzle)
        self.data = list(self.data)
        self.test_data = list(self.test_data)
        self.vocab = buildVocab(self.data + self.test_data)
            
    def train(self):
        self.generateData()
        model = TwoLayerClassifier(self.num_choices, 
                                   self.num_choices * len(self.vocab), 
                                   self.hidden_layer_size)
        print(model)
        loss_function = nn.NLLLoss()
        #optimizer = optim.SGD(model.parameters(), lr=0.1)
        optimizer = optim.Adam(model.parameters())
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
            train_acc = self.evaluate(model, self.data)
            test_acc = self.evaluate(model, self.test_data)
            print('train: {:.2f}; test: {:.2f}'.format(train_acc, test_acc))
        return model


    def batchTrain(self, batch_size):
        self.generateData()
        model = TwoLayerClassifier(self.num_choices, 
                                   self.num_choices * len(self.vocab), 
                                   self.hidden_layer_size)
        cudaify(model)
        print(model)
        loss_function = nn.NLLLoss()
        #optimizer = optim.SGD(model.parameters(), lr=0.1)
        optimizer = optim.Adam(model.parameters())
        for epoch in range(self.num_training_epochs):
            model.zero_grad()
            batch = random.sample(self.data, batch_size)
            input_matrix = makePuzzleMatrix(batch, self.vocab)
            target = makePuzzleTargets([label for (_, label) in batch])
            log_probs = model(input_matrix)
            loss = loss_function(log_probs, target)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print('epoch {}'.format(epoch))
                train_acc = self.evaluate(model, self.data[:200])
                test_acc = self.evaluate(model, self.test_data)
                print('train: {:.2f}; test: {:.2f}'.format(train_acc, test_acc))
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



NUM_CHOICES = 5
BATCH_SIZE = 3000

BASE_10 = 'ABCDEFGHIJK'
BASE_26 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
BASE_36 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
BASE_62 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz'

trainer = Trainer(AltTwoDigitPuzzleGenerator(BASE_10, NUM_CHOICES), NUM_CHOICES)
model = trainer.batchTrain(BATCH_SIZE)    
print('training accuracy = {}'.format(trainer.evaluate(model, trainer.data[:500])))
print('test accuracy = {}'.format(trainer.evaluate(model, trainer.test_data)))