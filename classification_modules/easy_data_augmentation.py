import numpy as np
import pandas as pd

def random_swap(sentence : list, p : float):
    """
    Function performs random swap i.e. randomly choses two words in the sentence and swap their positions. Do this n times.

    Args:
        sentence - input sentence
        p - probability of word deletion in random_deletion()
    """
    l = len(sentence)
    n = int(p) * l
    sen = sentence.copy()

    for i in range(n):
        w1 = np.random.choice(sen)
        w2 = np.random.choice(sen.remove(w1))

        w1_index = sentence.index(w1)
        w2_index = sentence.index(w2)

        sentence[w1_index] = w2
        sentence[w2_index] = w1
    
    return sentence

def random_deletion(sentence : list, p : float):
    """
    Function performs random deletion i.e. randomly removes each word in the sentence with probability p.

    Args:
        sentence - input sentence
        p - probability of word deletion
    """
    for w in sentence:
        if np.random.choice([1,0], p=[p, 1-p])==1:
            sentence.remove(w)
        else:
            continue

    return sentence

def perform_eda(df : pd.DataFrame, p : float, N : int):
    """
    Function performs 2 of easy data augmentation techniques: random swap (RS) and random deletion (RA). 

    Args:
        df - dataframe on which eda should be performed
        p - probability of word deletion in RA
        N - number of augmented samples to be added
    """
    df['eda'] = 0
    i = 0

    train_eda = df[['text_preprocessed', 'label', 'eda']].copy()

    while i < N:
        
        sentence_eda = []
        edas = []
        labels = []
        
        for text in train_eda[train_eda.label == 1].text_preprocessed.values:
            
            try:
                sentence = text.strip('][').split(', ')
            except:
                sentence = text

            if np.random.choice([1,0]) == 1:
                sentence_eda.append(random_swap(sentence, p))
            else:
                sentence_eda.append(random_deletion(sentence, p))
                
            edas.append(0)
            labels.append(1)

            i += 1
            if i == N:
                break
        
        train_eda = pd.concat([train_eda, pd.DataFrame({'text_preprocessed' : sentence_eda, 'label' : labels, 'eda' : edas})])

    return train_eda