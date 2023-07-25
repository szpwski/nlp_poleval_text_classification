import re
import pandas as pd
import string
import spacy
nlp = spacy.load('pl_core_news_sm')

def encoded_special_signs(text : str, clean : bool = False):
    """
    Function checks if text contains emojis and how many

    Args:
        text - text to check
        clean - if emojis should be cleaned
    """

    # Define polish letters
    pl_letters = 'ą ć ę ł ń ó ś ź ż'
    
    # Create list of special signs in text
    special_signs = [l for l in text.lower() if l not in list(string.ascii_lowercase) and l not in set(string.punctuation) and l not in [' ', '  '] and l.isdigit() == False]
    
    # Create list of encoded special signs
    special_signs_encoded = [l.encode() for l in special_signs]
    
    # Create list of True/False values checking if given special sign is emoji
    is_emoji = [l not in pl_letters.encode().split() for l in special_signs_encoded]

    if clean:
        
        emojis = list(pd.Series(special_signs)[is_emoji])
        
        for e in emojis:
            text = text.replace(e, '')

        return sum(is_emoji), text
    
    return sum(is_emoji)

def clean_text(text, punctuation=True):
    """
    Function clean texts from special signs, emojis and mentions
    
    Args:
        text - text to be cleaned
        punctuation - if punctuation should be removed
    """

    # Remove emojis
    text = encoded_special_signs(text, clean=True)[1]

    # Remove mentions
    text = text.replace('@anonymized_account', '')

    if punctuation:
        # Remove special characters
        for p in set(string.punctuation):
            
            text = text.replace(p, ' ')

    text = text.replace('\\n', ' ')

    # Remove double spaces
    text = re.sub(' +', ' ', text)

    # Remove forward and backward spaces
    text = text.strip()

    return text


def is_token_allowed(token):
    """
    Function checks whether token is stop word or punctuation

    Args:
        token - nlp spacy token
    """
    return bool(
         token
         and str(token).strip()
         and not token.is_stop
         and not token.is_punct
         )

def preprocess_token(token):
    """
    Function preprocesses token by applying lemmatization, strip and normalization (lowercase)

    Args:
        token - nlp spacy token
    """
    return token.lemma_.strip().lower()