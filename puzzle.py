import random

import torch

torch.manual_seed(1)


def oneHot(word, vocab):
    vec = [0]*len(vocab)
    vec[vocab[word]] = 1
    return vec

def makePuzzleVector(puzzle, vocab):
    (num1, num2, num3), _ = puzzle
    oneHot1 = oneHot(str(num1), vocab)
    oneHot2 = oneHot(str(num2), vocab)
    oneHot3 = oneHot(str(num3), vocab)
    return torch.FloatTensor(oneHot1 + oneHot2 + oneHot3).view(1, -1)


def makePuzzleTarget(label):
    return torch.LongTensor([label])    

def buildVocab(puzzles):
    word_to_ix = {}
    for choices, _ in puzzles:
        for word in choices:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    return word_to_ix

def flipCoin():
    return random.random() < 0.5
    

class OddManOutPuzzleGenerator:
    def __init__(self):
        pass
    
    def getTrainingData(self, N):    
        return [self.generate() for n in range(N)]


class TwoDigitPuzzleGenerator(OddManOutPuzzleGenerator):
    
    def __init__(self, digits):
        super(TwoDigitPuzzleGenerator, self).__init__()
        self.digits = digits

    def makeTwoDigitNumber(self, digit1, digit2):
        return digit1 * 10 + digit2
        

    def generate(self):
        digits = self.digits
        random.shuffle(digits)
        num1 = self.makeTwoDigitNumber(digits[0], digits[1])
        num2 = self.makeTwoDigitNumber(digits[0], digits[2])
        num3 = self.makeTwoDigitNumber(digits[3], digits[4])
        puzzle = [(str(num1), 0), (str(num2), 0), (str(num3), 1)]
        random.shuffle(puzzle)
        xyz = [i for (i,_) in puzzle]
        onehot = [j for (_,j) in puzzle]
        return (xyz, onehot.index(1))
    
    
class InvertingTwoDigitPuzzleGenerator(TwoDigitPuzzleGenerator):
    
    def makeTwoDigitNumber(self, digit1, digit2):
        num = digit1 * 10 + digit2
        if flipCoin():
            num = digit2 * 10 + digit1
        return num
