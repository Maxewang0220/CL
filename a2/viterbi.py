# # viterbi algorithm
def viterbi(init_prob: dict, transition_prob: dict, emission_prob: dict, input_sequence: list):
    back_pointer_list = []
    v_list = []

    # # calculate init probs
    v = dict()

    for state in init_prob:
        v[state] = init_prob[state] * emission_prob[state][input_sequence[0]]

    # # store the first v and state
    v_list.append(v)

    # # iteratively calculate the other v of the sequence
    for i in range(1, len(input_sequence)):
        v = dict()
        last_v = v_list[-1]
        obs = input_sequence[i]
        back_pointer = {}

        for current_state in init_prob:
            v[current_state], back_pointer[current_state] = max(
                [(last_v[last_state] * transition_prob[last_state][current_state] *
                  emission_prob[current_state][obs], last_state) for last_state in
                 init_prob])

        # # store current v and back pointer
        v_list.append(v)
        back_pointer_list.append(back_pointer)

    # # reconstruct the best path sequence
    best_path = []
    max_likelihood, last_state = max((v_list[-1][state], state) for state in init_prob)
    best_path.append(last_state)

    for i in range(len(input_sequence) - 2, -1, -1):
        last_state = back_pointer_list[i][last_state]
        best_path.append(last_state)

    best_path.reverse()

    return max_likelihood, best_path


if __name__ == "__main__":
    test_init_prob = {"H": 0.8, "C": 0.2}
    test_transition_prob = {"H": {"H": 0.7, "C": 0.3}, "C": {"H": 0.4, "C": 0.6}}
    test_emission_prob = {"H": {"1": 0.2, "2": 0.4, "3": 0.4}, "C": {"1": 0.5, "2": 0.4, "3": 0.1}}
    input_sequence1 = ["3", "1", "3"]
    input_sequence2 = ["1", "1", "1"]
    input_sequence3 = ["1"]

    print(input_sequence1, viterbi(test_init_prob, test_transition_prob, test_emission_prob, input_sequence1))
    print(input_sequence2, viterbi(test_init_prob, test_transition_prob, test_emission_prob, input_sequence2))
    print(input_sequence3, viterbi(test_init_prob, test_transition_prob, test_emission_prob, input_sequence3))
