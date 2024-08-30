from viterbi import viterbi

if __name__ == '__main__':
    input_sequence = {}
    prob, pos = viterbi(input_sequence, init_prob, transition_prob, emission_prob)

