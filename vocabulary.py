import nltk
import codecs
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from googletrans import Translator


translator = Translator()


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
        translated_text = translator.translate(raw_text, src=lang_suppose, dest=TARGET_LANG)
        return translated_text


filename = 'Dracula'
income_file = f'{filename}.txt'
alien_text = Text.initiate_text(income_file)


def sentence_tokenization():
    raw_text = alien_text
    sent_token = nltk.sent_tokenize(raw_text)
    return sent_token


def word_sentence_tokenization():
    sent_token = sentence_tokenization()
    word_sent_tokens = [nltk.word_tokenize(token) for token in sent_token]
    return word_sent_tokens


def word_tokenization():
    raw_text = alien_text
    word_token = nltk.word_tokenize(raw_text)
    return word_token


def create_unique_set():
    unique_tokens = set(word_tokenization())
    print(unique_tokens)


def save_to_file():
    with open(f'{filename}_sent_tokens.txt', 'w', encoding="utf-8") as file:
        for token in sent_token:
            file.write(token + '\n')


if __name__ == '__main__':
    word_sentence_tokenization()
