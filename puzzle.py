import random

import torch

torch.manual_seed(1)




def flipCoin():
    return random.random() < 0.5
    

class PuzzleGenerator:
    def __init__(self):
        pass
    
    def getTrainingData(self, N):    
        return [self.generate() for n in range(N)]


class TwoDigitPuzzleGenerator(PuzzleGenerator):
    
    def __init__(self, digits):
        super(PuzzleGenerator, self).__init__()
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
    
class OddManOutPuzzleGenerator(PuzzleGenerator):
    
    def __init__(self, buckets, choicesPerPuzzle):
        self.buckets = dict()
        # Remove any buckets that don't contains enough choices
        for bucket in buckets:
            if len(buckets[bucket]) >= choicesPerPuzzle - 1:
               self.buckets[bucket] = buckets[bucket] 
        if len(self.buckets) < 2:
            raise Exception("""Argument 'buckets' must have 
                            at least 2 keys: {}""".format(buckets))
        self.choicesPerPuzzle = choicesPerPuzzle
 
    def getCandidatePuzzle(self):
        categories = list(self.buckets.keys())
        bucket = random.choice(categories)
        otherBucket = random.choice(categories)            
        correct = random.sample(self.buckets[bucket], self.choicesPerPuzzle - 1)
        oddman = random.choice(self.buckets[otherBucket])
        return (correct, oddman)

    def countCommonCategories(self, choices):
        return len([bucket for bucket in self.buckets 
                    if choices.issubset(set(self.buckets[bucket]))])
        

    def findOddmen(self, choices):
        choiceset = set(choices)
        num_common = self.countCommonCategories(choiceset)
        oddmen = []
        for i, oddman in enumerate(choices):
            remainder = choiceset - set([oddman])
            if self.countCommonCategories(remainder) > num_common:
                oddmen.append(i)
        return oddmen
            
    
    def generate(self):
        (correct, oddman) = self.getCandidatePuzzle()
        while len(self.findOddmen(correct + [oddman])) != 1:
            (correct, oddman) = self.getCandidatePuzzle()            
        choices = correct + [oddman]
        random.shuffle(choices)
        answer = self.findOddmen(choices)[0]
        return (tuple(choices), answer)
    
class AltTwoDigitPuzzleGenerator(PuzzleGenerator):
    
    def __init__(self, digits, choicesPerPuzzle):
        buckets = dict()
        for digit1 in digits:
            bucket = set()
            for digit2 in digits:
                bucket.add(str(digit1) + str(digit2))
                bucket.add(str(digit2) + str(digit1))
            buckets[str(digit1)] = list(bucket)
        self.generator = OddManOutPuzzleGenerator(buckets, choicesPerPuzzle)
        
    def generate(self):
        return self.generator.generate()
            

class ThreeDigitPuzzleGenerator(PuzzleGenerator):
    
    def __init__(self, digits, choicesPerPuzzle):
        buckets = dict()
        for digit1 in digits:
            for digit2 in digits:
                bucket1 = set()
                bucket2 = set()
                bucket3 = set()
                for digit3 in digits:
                    bucket3.add(str(digit1) + str(digit2) + str(digit3))
                    bucket2.add(str(digit1) + str(digit3) + str(digit2))
                    bucket1.add(str(digit3) + str(digit1) + str(digit2))    
                buckets['*' + str(digit2) + str(digit3)] = list(bucket1)
                buckets[str(digit1) + '*' + str(digit2)] = list(bucket2)
                buckets[str(digit1) + str(digit2) + '*'] = list(bucket3)
        self.generator = OddManOutPuzzleGenerator(buckets, choicesPerPuzzle)
        
    def generate(self):
        return self.generator.generate()
 
       
        
