from nltk.probability import (FreqDist, ConditionalFreqDist, ConditionalProbDist, MLEProbDist, SimpleGoodTuringProbDist)
from nltk.util import ngrams


def ml_estimator(freqdist):
    return MLEProbDist(freqdist)

def goodturing_estimator(freqdist):
    return SimpleGoodTuringProbDist(freqdist)

class BasicNgram(ConditionalProbDist):
    """
    Define and train an Ngram Model over the corpus represented by the list words. 
    Given an BasicNgram instance ngram and a (n-1)-gram context (i.e., a tuple of n-1 strings), 
    a call to ngram[context] returns a nltk.probability.ProbDistI object representing the Probability distribution P(.|context) over possible values for the next word. 
    Be aware that context has to be a tuple, even if context is a unigram (see example below)
    
    >>> corpus=['a','b','b','a']
    >>> bigram=BasicNgram(2,corpus)
    >>> bigram.contexts()
    [('<$>',), ('a',), ('b',)]
    >>> p_b=bigram[('b',)] #not bigram['b']!!!
    >>> p_b.prob('a')
    0.5
    >>> p_b.prob('b')
    0.5
    
    :param n: the dimension of the n-grams (i.e. the size of the context+1).
    :type n: int
    :param corpus: 
    :type corpus: list(Str)
    
    other parameters are optional and may be omitted. They define whether to add artificial symbols before or after the word list, 
    and whether to use another estimation methods than maximum likelihood.
    """
    def __init__(self, n, words, start_symbol="<$>", end_symbol="</$>", pad_left=True, pad_right=False, estimator=ml_estimator):
        assert (n > 0)
        self._n=n
        self._words=words
        self._counter=ConditionalFreqDist()
        self._start_symbol=start_symbol
        self._end_symbol=end_symbol
        self._pad_left=pad_left
        self._pad_right=pad_right
        self._train()
        super().__init__(self._counter, estimator)
        
        
    def _train(self):       
        _ngrams=self.generate_ngrams()        
        for ngram in _ngrams:
            context=ngram[0:-1]
            outcome=ngram[-1]
            self._counter[context][outcome]+=1
            
    """
    returns an iterable over the ngrams of the word corpus
    """
    def generate_ngrams(self):
        return ngrams(self._words, self._n, self._pad_left, self._pad_right,
                      left_pad_symbol=self._start_symbol,
                      right_pad_symbol=self._end_symbol)


    """                                                                                                                                                                                                                                                                                                                                                               
    Return the list of contexts                                                                                                                                                                                                                                                                                                                                       
    """
    def contexts(self):
        return list(self.conditions())
            
        
