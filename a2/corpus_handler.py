import conllu
from conllu import parse_incr


class CorpusHandler:
    file_path = None

    def __init__(self, file_path):
        self.file_path = file_path

    def train_on_corpus(self):
        # open corpus file and train HMM
        with open(self.file_path, 'r', encoding='utf-8') as file:
            reader = parse_incr(file)
            init_prob, transition_prob, emission_prob = self.train(reader)

        return init_prob, transition_prob, emission_prob

    def write_corpus(self):
        pass

    def train(self, reader):
        init_prob = dict()
        transition_prob = dict()
        emission_prob = dict()

        sentence_count = 0
        first_pos_count = dict()
        pos_count = dict()
        transition_count = dict()
        emission_count = dict()

        for sentence in reader:
            sentence_count += 1
            first_pos = sentence[0]['upos']

            if first_pos not in first_pos_count:
                first_pos_count[first_pos] = 1
            else:
                first_pos_count[first_pos] += 1

            for i in range(len(sentence)):
                pos = sentence[i]['upos']
                word = sentence[i]['form']

                # # count pos
                if pos not in pos_count:
                    pos_count[pos] = 1
                else:
                    pos_count[pos] += 1

                # # count transition
                if i > 0:
                    last_pos = sentence[i - 1]['upos']
                    if last_pos not in transition_count:
                        transition_count[last_pos] = dict()
                        transition_count[last_pos][pos] = 1
                    else:
                        if pos not in transition_count[last_pos]:
                            transition_count[last_pos][pos] = 1
                        else:
                            transition_count[last_pos][pos] += 1

                # # count emission
                if pos not in emission_count:
                    emission_count[pos] = dict()
                    emission_count[pos][word] = 1
                else:
                    if word not in emission_count[pos]:
                        emission_count[pos][word] = 1
                    else:
                        emission_count[pos][word] += 1

        # # calculate init prob
        for pos in pos_count:
            if pos not in first_pos_count:
                init_prob[pos] = 0
            else:
                init_prob[pos] = first_pos_count[pos] / sentence_count

        # # calculate transition prob
        for last_pos in transition_count:
            transition_prob[last_pos] = dict()
            for current_pos in transition_count[last_pos]:
                # # add smoothing
                # transition_prob[last_pos][current_pos] = (transition_count[last_pos][current_pos] + 1) / (
                #         pos_count[last_pos] + len(pos_count))
                transition_prob[last_pos][current_pos] = (transition_count[last_pos][current_pos]) / (
                    pos_count[last_pos])

        # # calculate emission prob
        for pos in emission_count:
            emission_prob[pos] = dict()
            for word in emission_count[pos]:
                # # add smoothing
                emission_prob[pos][word] = (emission_count[pos][word] + 1) / (pos_count[pos] + len(pos_count))

        return init_prob, transition_prob, emission_prob


if __name__ == "__main__":
    corpusReader = CorpusHandler("./data/de_gsd-ud-test.conllu")
    init_prob, transition_prob, emission_prob = corpusReader.train_on_corpus()
    print(init_prob)
    print(transition_prob)
