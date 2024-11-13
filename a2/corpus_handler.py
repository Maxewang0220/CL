import numpy as np
from conllu import parse_incr
import matplotlib.pyplot as plt
from viterbi import viterbi
import time
from scipy.interpolate import interp1d


# process the corpus and train the HMM model
class CorpusHandler:
    file_path = None
    init_prob = None
    transition_prob = None
    emission_prob = None

    def __init__(self, file_path):
        self.file_path = file_path

    def train_on_corpus(self, sentence_num):
        # open corpus file and train HMM
        with open(self.file_path, 'r', encoding='utf-8') as file:
            reader = parse_incr(file)
            init_prob, transition_prob, emission_prob = self.train(reader, sentence_num)

        return init_prob, transition_prob, emission_prob

    def train(self, reader, sentence_num):
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

            if sentence_count > sentence_num:
                break

            # count the first word pos for init prob
            first_pos = sentence[0]['upos']

            if first_pos not in first_pos_count:
                first_pos_count[first_pos] = 1
            else:
                first_pos_count[first_pos] += 1

            for i in range(len(sentence)):
                pos = sentence[i]['upos']
                word = sentence[i]['form']

                # count pos
                if pos not in pos_count:
                    pos_count[pos] = 1
                else:
                    pos_count[pos] += 1

                # count transition
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

        # calculate init prob for each pos (init_prob contains every pos)
        for pos in pos_count:
            if pos not in first_pos_count:
                init_prob[pos] = 0
            else:
                init_prob[pos] = first_pos_count[pos] / sentence_count

        # calculate transition prob
        for last_pos in transition_count:
            transition_prob[last_pos] = dict()
            for current_pos in init_prob:
                # add Laplace smoothing
                if current_pos not in transition_count[last_pos]:
                    transition_prob[last_pos][current_pos] = 1 / (pos_count[last_pos] + len(init_prob))
                else:
                    transition_prob[last_pos][current_pos] = (transition_count[last_pos][current_pos] + 1) / (
                            pos_count[last_pos] + len(init_prob))

        # calculate emission prob
        for pos in emission_count:
            emission_prob[pos] = dict()
            for word in emission_count[pos]:
                # add Laplace smoothing
                # although word in emission_count is not zero, Laplace smoothing is still useful as it can improve the prob
                # balance between the major words and minor words
                emission_prob[pos][word] = (emission_count[pos][word] + 1) / (pos_count[pos] + len(emission_count[pos]))

        # store train result
        self.init_prob = init_prob
        self.transition_prob = transition_prob
        self.emission_prob = emission_prob

        return init_prob, transition_prob, emission_prob

    # predict
    def predict(self, evaluate_file_path):
        token_count = 0
        accurate_pos_count = 0

        # open evaluation corpus file and predict pos
        with open(evaluate_file_path, 'r', encoding='utf-8') as file:
            reader = parse_incr(file)
            for sentence in reader:
                # predict token pos
                token_list = [token['form'] for token in sentence]
                token_count += len(token_list)
                _, pos_list = viterbi(self.init_prob, self.transition_prob, self.emission_prob, token_list)

                # compare predict pos and gold pos
                for i in range(len(sentence)):
                    gold_pos = sentence[i]['upos']
                    predict_pos = pos_list[i]
                    if predict_pos == gold_pos:
                        accurate_pos_count += 1

        # calculate accuracy
        accuracy = accurate_pos_count / token_count

        return accuracy

    # Extra: show the speed vs. sentence length curve
    def predict_time(self, evaluate_file_path_list):

        sentence_time_cost = [0] * 150
        sentence_length_count = [0] * 150

        for evaluate_file_path in evaluate_file_path_list:
            # open evaluation corpus file and predict pos
            with open(evaluate_file_path, 'r', encoding='utf-8') as file:
                reader = parse_incr(file)
                for sentence in reader:
                    # predict token pos
                    token_list = [token['form'] for token in sentence]
                    sentence_length = len(sentence)

                    # time count
                    strart_time = time.time()
                    _, pos_list = viterbi(self.init_prob, self.transition_prob, self.emission_prob, token_list)
                    finish_time = time.time()

                    # save the time cost
                    sentence_time_cost[sentence_length] += finish_time - strart_time
                    sentence_length_count[sentence_length] += 1

        # calculate average time cost
        time_length = [sentence_time_cost[sentence_length] * 1000 / sentence_length_count[sentence_length]
                       if sentence_length_count[sentence_length] != 0 else None for sentence_length in range(150)]

        return time_length


if __name__ == "__main__":
    # define file path
    DE_GSD_TRAIN = "./data/de_gsd-ud-train.conllu"
    DE_GSD_TEST = "./data/de_gsd-ud-test.conllu"
    DE_SGD_DEV = "./data/de_gsd-ud-dev.conllu"

    corpusHandler = CorpusHandler(DE_GSD_TRAIN)

    test_accuracies = []
    dev_accuracies = []

    # change this variable to adjust the increment of training data size
    size_increment = 500

    time1 = time.time()
    for init_sentence_num in range(500, 14500, size_increment):
        init_prob, transition_prob, emission_prob = corpusHandler.train_on_corpus(init_sentence_num)
        test_accuracy = corpusHandler.predict(DE_GSD_TEST)
        dev_accuracy = corpusHandler.predict(DE_SGD_DEV)
        print("test accruacy", test_accuracy)
        print("dev accruacy", dev_accuracy)
        test_accuracies.append(test_accuracy)
        dev_accuracies.append(dev_accuracy)

    # Extra: show the accuracy-train data size plot
    plt.plot(range(500, 14500, size_increment), test_accuracies, label='Test Accuracy')
    plt.plot(range(500, 14500, size_increment), dev_accuracies, label='Dev Accuracy')
    plt.xlabel('Training Data Size')
    plt.ylabel('Accuracy')
    plt.title('Accuracy-Training Data Size')
    plt.legend()
    plt.show()

    print(time.time() - time1)

    # Extra: show the speed vs. sentence length curve
    time_length = corpusHandler.predict_time([DE_GSD_TRAIN, DE_GSD_TEST, DE_SGD_DEV])
    # get the index and value of non-None values
    x = [i for i, value in enumerate(time_length) if value is not None]
    y = [value for value in time_length if value is not None]

    # generate interpolation function (cubic spline interpolation)
    interpolator = interp1d(x, y, kind='cubic')

    # generate more dense points for smooth curve
    x_new = np.linspace(min(x), max(x), num=100)
    y_smooth = interpolator(x_new)

    # show the curve
    plt.plot(x_new, y_smooth, label="Smooth Curve")
    plt.scatter(x, y, color="red", label="Original Points")  # 用红点标出原始数据点
    plt.xlabel("Sentence Length")
    plt.ylabel("Time Cost")
    plt.legend()
    plt.title("Speed vs. Sentence Length Curve")
    plt.show()

    # # test if init prob/ transition prob/ emission prob is correct
    # print(init_prob)
    # print(transition_prob)
    # print(emission_prob)

    # # test if viterbi alg is correct
    # print(viterbi(init_prob, transition_prob, emission_prob,
    #               ['Der', 'Hauptgang', 'war', 'in', 'Ordnung', ',', 'aber', 'alles', 'andere', 'als', 'umwerfend',
    #                '.']))
    #
    # print(viterbi(init_prob, transition_prob, emission_prob,
    #               ['Anders', 'kann', 'ich', 'es', 'nicht', 'ausdrücken', '.', ]))
    #
    # print(viterbi(init_prob, transition_prob, emission_prob,
    #               ['Bester',
    #                'Kaffee',
    #                'im',
    #                'Veedel',
    #                'sowieso',
    #                '.']
    #               ))
