"""
import nltk
nltk.download('wordnet')
"""

from nltk.corpus import wordnet as wn

def synsets_of_mercury():
    synsets = wn.synsets('mercury')
    for synset in synsets:
        print(synset)
        

def hypernym_chain(synset_name):
    """
    e.g. hypernym_chain('boat.n.01')
    
    """
    synset = wn.synset(synset_name)
    result = [synset]
    while len(synset.hypernyms()) > 0:
        synset = synset.hypernyms()[0]
        result.append(synset)
    return result

def get_all_hypernyms_from_sense(sense):
    result = set()
    for y in sense.hypernyms():
        result.add(y)
        for z in get_all_hypernyms_from_sense(y):
            result.add(z)
    return result

def get_all_hypernyms(word):
    """
    e.g. get_all_hypernyms('dog')
    
    """
    result = set()
    for x in wn.synsets(word):
        for y in get_all_hypernyms_from_sense(x):
            result.add(y)
    return result


def get_all_hyponyms_from_sense(sense):
    """
    e.g. get_all_hyponyms_from_sense(wn.synset('metallic_element.n.01'))
    
    """
    result = set()
    for y in sense.hyponyms():
        result.add(y)
        for z in get_all_hyponyms_from_sense(y):
            result.add(z)
    return result


class Specificity:
    def __init__(self):
        self.cache = dict()
        
    def evaluate(self, sense):
        if sense.name() not in self.cache:
            spec = len(get_all_hyponyms_from_sense(sense))
            self.cache[sense.name()] = spec
        return self.cache[sense.name()]


specificity = Specificity()


def find_lowest_common_ancestor(words):
    """
    find_lowest_common_ancestor(['libertarian', 'green', 'garden', 'democratic'])
    find_lowest_common_ancestor(['apple', 'banana', 'orange', 'grape'])
    
    """
    commonHypernyms = get_all_hypernyms(words[0])
    for word in words[1:]:
        commonHypernyms = commonHypernyms & get_all_hypernyms(word)
    if len(commonHypernyms) == 0:
        hyp = wn.synset('entity.n.01')
        return (specificity.evaluate(hyp), hyp)
    scoredHypernyms = [(specificity.evaluate(hyp), hyp) for hyp in commonHypernyms]
    sortedHypernyms = sorted(scoredHypernyms)
    return sortedHypernyms[0]





                
