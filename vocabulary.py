import nltk
import codecs
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from googletrans import Translator


class Text:
    def initiate_text(filename):
        with codecs.open(f'{filename}', 'r', 'utf_8_sig') as text_source:
            raw_text = " ".join(text_source.readlines())
        return raw_text

    def language_detect(raw_text):
        lang_suppose = detect(raw_text)
        return lang_suppose

    def translate(raw_text, lang_suppose, dest=None):
        TARGET_LANG = 'en'
        translated_text = Translator.translate(raw_text, src=lang_suppose, dest=TARGET_LANG)
        return translated_text


filename = 'Rapunzel'
income_file = f'{filename}.txt'
alien_text = Text.initiate_text(income_file)


def word_tokenization():
    text = str(alien_text)
    word_token = nltk.word_tokenize(text)
    return word_token


if __name__ == '__main__':
    ...
