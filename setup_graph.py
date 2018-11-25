"""Author: Brandon Trabucco, Copyright 2019
Implements a proprocessor for converting a sentence into a binary tree.
"""


import numpy as np
from detailed_captioning.utils import load_glove
from detailed_captioning.utils import get_pos_tagger


if __name__ == "__main__":
    
    tagger = get_pos_tagger()
    word_vocabulary, word_embeddings = load_glove()
    tagged_vocabulary = list(zip(*tagger.tag(word_vocabulary.reverse_vocab)))[1]
    
    sentence = "a black and white spotted cat sleeping on a sofa cushion ."
    tokens = sentence.split(" ")
    
    POS_scores = {"NOUN": 11, "VERB": 10, "ADJ": 9, "NUM": 8,
        "ADV": 7, "PRON": 6, "PRT": 5, "ADP": 4,
        "DET": 3, "CONJ": 2, ".": 1, "X": 0 }
    
    def get_scores(tokens):
        ids = [word_vocabulary.word_to_id(w) for w in tokens]
        tags = [POS_scores[tagged_vocabulary[i]] for i in ids]
        scores = [np.log(i + np.exp(t))  for i, t in zip(ids, tags)]
        return scores
    
    scores = get_scores(tokens)
    print(list(zip(tokens, scores)))

