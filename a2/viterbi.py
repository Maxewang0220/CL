# viterbi algorithm
def viterbi(init_prob: dict, transition_prob: dict, emission_prob: dict, input_sequence: list):
    """
    :param init_prob: Dict of initial probabilities for each state (contains every state)
    :param transition_prob: Dict of transition probabilities between states
    :param emission_prob: Dict of emission probabilities for each state-observation pair
    :param input_sequence: List of observations
    :return: The maximum likelihood and the best path sequence list
    """

    # check if the input sequence is empty
    if len(input_sequence) < 1:
        return None

    # create lists to store the v and back pointer
    back_pointer_list = []
    v_list = []

    # calculate init probs
    v = dict()

    # check if the word exists
    occur_flag = False
    for state in init_prob:
        if input_sequence[0] not in emission_prob[state]:
            v[state] = 0
        else:
            occur_flag = True
            v[state] = init_prob[state] * emission_prob[state][input_sequence[0]]

    # if the word doesn't exist then ignore its emission prob
    if not occur_flag:
        for state in v:
            v[state] = init_prob[state]

    # # store the first v
    v_list.append(v)

    # iteratively calculate the other v of the sequence
    for i in range(1, len(input_sequence)):
        v = dict()
        last_v = v_list[-1]
        obs = input_sequence[i]
        back_pointer = {}

        # check word
        occur_flag = False
        for current_state in init_prob:
            if obs not in emission_prob[current_state]:
                emission_prob[current_state][obs] = 0
            else:
                occur_flag = True

        # set emission_prob = 1 for all states if the word is unknown
        if not occur_flag:
            for current_state in init_prob:
                emission_prob[current_state][obs] = 1

        for current_state in init_prob:
            v[current_state], back_pointer[current_state] = max(
                (last_v[last_state] * transition_prob[last_state][current_state] *
                 emission_prob[current_state][obs], last_state) for last_state in
                init_prob)

        # store current v and back pointer
        v_list.append(v)
        back_pointer_list.append(back_pointer)

    # reconstruct the best path sequence
    best_path = []
    max_likelihood, last_state = max((v_list[-1][state], state) for state in init_prob)
    best_path.append(last_state)

    for i in range(len(input_sequence) - 2, -1, -1):
        last_state = back_pointer_list[i][last_state]
        best_path.append(last_state)

    best_path.reverse()

    return max_likelihood, best_path


if __name__ == "__main__":
    # test viterbi alg
    test_init_prob = {"H": 0.8, "C": 0.2}
    test_transition_prob = {"H": {"H": 0.7, "C": 0.3}, "C": {"H": 0.4, "C": 0.6}}
    test_emission_prob = {"H": {"1": 0.2, "2": 0.4, "3": 0.4}, "C": {"1": 0.5, "2": 0.4, "3": 0.1}}

    input_sequence1 = ["3", "1", "3"]
    input_sequence2 = ["1", "1", "1"]
    input_sequence3 = ["1"]

    print(input_sequence1, viterbi(test_init_prob, test_transition_prob, test_emission_prob, input_sequence1))
    print(input_sequence2, viterbi(test_init_prob, test_transition_prob, test_emission_prob, input_sequence2))
    print(input_sequence3, viterbi(test_init_prob, test_transition_prob, test_emission_prob, input_sequence3))

    test_init_prob = {'DT': 0.8, 'NN': 0.2, 'VB': 0}
    test_transition_prob = {"DT": {"DT": 0, "NN": 0.9, "VB": 0.1}, "NN": {"DT": 0, "NN": 0.5, "VB": 0.5},
                            "VB": {"DT": 0.5, "NN": 0.5, "VB": 0}}
    test_emission_prob = {"DT": {"the": 0.2, "fans": 0, "watch": 0, "race": 0},
                          "NN": {"the": 0, "fans": 0.1, "watch": 0.3, "race": 0.1},
                          "VB": {"the": 0, "fans": 0.2, "watch": 0.15, "race": 0.3}}
    input_sequence4 = ["the", "fans", "watch", "the", "race"]
    print(input_sequence4, viterbi(test_init_prob, test_transition_prob, test_emission_prob, input_sequence4))
