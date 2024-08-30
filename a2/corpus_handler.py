import conllu
from conllu import parse_incr


class CorpusReader:
    reader = None

    def __init__(self, file_path):
        self.reader = self.read_corpus(file_path=file_path)

    def read_corpus(self, file_path):
        # open corpus file
        with open(file_path, 'r') as file:
            reader = parse_incr(file)

        return reader


class CorpusWriter:
    pass


if __name__ == "__main__":
    corpusReader = CorpusReader("./data/de_gsd-ud-dev.conllu")
    