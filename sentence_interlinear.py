import nltk
from googletrans import Translator

from vocabulary import Text


filename = 'Rapunzel'
income_file = f'{filename}.txt'
alien_text = Text.initiate_text(income_file)


def translation():
    ...


def sentence_tokenisation():
    raw_text = alien_text
    sent_token = nltk.sent_tokenize(raw_text)
    with open(f'{filename}_sent_tokens.txt', 'w', encoding="utf-8") as file:
        for token in sent_token:
            file.write(token + '\n')


def compose_interlinear(income_file, translated_file):
    with open(income_file, 'r') as file1, \
         open(translated_file, 'r') as file2, \
         open(f'{filename}_interlinear.txt', 'w') as file3:
        file3.write(f'The file was translated automatically.\n'
                    f'Source language is: {Text.language_detect(alien_text)}\n'
                    f'Translation language is English.\n')
        for p in zip(file1, file2):
            print(*map(lambda s: s.strip(), p), sep='\n', file=file3)


if __name__ == '__main__':
    sentence_tokenisation()
