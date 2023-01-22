import nltk
import codecs
from langdetect import detect


class Text:
    def initiate_text(filename):
        with codecs.open(f'{filename}', 'r', 'utf_8_sig') as text_source:
            raw_text = " ".join(text_source.readlines())
        return raw_text

    def language_detect(raw_text):
        lang_suppose = detect(raw_text)
        return lang_suppose


filename = 'Rapunzel'
income_file = f'{filename}.txt'
alien_text = Text.initiate_text(income_file)


def sentence_tokenization():
    text = str(alien_text)
    sent_token = nltk.sent_tokenize(text)
    return sent_token


def word_tokenization():
    text = str(alien_text)
    word_token = nltk.word_tokenize(text)
    return word_token


if __name__ == '__main__':
    ...
