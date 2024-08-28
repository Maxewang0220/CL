# Code for the assignment in https://github.com/coli-saar/cl/wiki/Assignment:-Dependency-parsing
# Alexander Koller, December 2023

def strip_none_heads(examples, i):
    tokens = examples["tokens"][i]
    heads = examples["head"][i]
    deprels = examples["deprel"][i]

    non_none = [(t, h, d) for t, h, d in zip(tokens, heads, deprels) if h != "None"]
    return zip(*non_none)

def map_first_occurrence(nums):
    """
    Maps a list of numbers to a dictionary that assigns each unique number the position of its first occurrence.

    Example:
    > map_first_occurrence([0,1,2,3,3,3,4])
    {0: 0, 1: 1, 2: 2, 3: 3, 4: 6}

    :param nums:
    :return:
    """
    seen = set()
    return {num: i for i, num in enumerate(nums) if num is not None and num not in seen and not seen.add(num)}


def pad_to_same_size(lists, padding_symbol):
    maxlen = max([len(l) for l in lists])
    return [l + (padding_symbol,)*(maxlen-len(l)) for l in lists]

def tokenize_and_align_labels(examples, skip_index=-100):
    # delete tokens with "None" head and their annotations
    examples_tokens, examples_heads, examples_deprels = [], [], []
    for sentence_id in range(len(examples["tokens"])):
        tt, hh, dd = strip_none_heads(examples, sentence_id)
        examples_tokens.append(tt)
        examples_heads.append(hh)
        examples_deprels.append(dd)

    tokenized_inputs = tokenizer(examples_tokens, truncation=True, is_split_into_words=True, padding=True) # get "tokenizer" from global variable
    # tokenized_inputs is a dictionary with keys input_ids and attention_mask;
    # each is a list (per sentence) of lists (per token).

    remapped_heads = [] # these will be lists (per sentence) of lists (per token)
    deprel_ids = []
    tokens_representing_words = []
    num_words : list[int] = []
    maxlen_t2w = 0  # max length of a token_to_word_here list

    for sentence_id, annotated_heads in enumerate(examples_heads):
        deprels = examples_deprels[sentence_id]
        word_ids = tokenized_inputs.word_ids(batch_index=sentence_id)
        word_pos_to_token_pos = map_first_occurrence(word_ids) # word-pos to first token-pos; both start at 0 for first word (actual) / first token (BOS)

        previous_word_idx = None
        heads_here : list[int] = []
        deprel_ids_here : list[int] = []

        # list of token positions that map to words (first token of each word)
        # token 0 -> word 0 (BOS)
        tokens_representing_word_here : list[int] = [0]

        for sentence_position, word_idx in enumerate(word_ids):
            # Special tokens (BOS, EOS) have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                heads_here.append(skip_index)
                deprel_ids_here.append(skip_index)

            # We set the label for the first token of each word;
            # subsequent tokens of the same word will have the same word_idx.
            elif word_idx != previous_word_idx:
                if annotated_heads[word_idx] == "None": # added by padding
                    print("A 'None' head survived!")
                    sys.exit(0)
                else:
                    # Map HEAD annotation to position of first token of head word.
                    # HEAD = 0 => map it to first token (BOS)
                    # Otherwise, look up first token for HEAD-1 (HEAD is 1-based, word positions are 0-based)
                    head_word_pos = int(annotated_heads[word_idx])
                    head_token_pos = 0 if head_word_pos == 0 else word_pos_to_token_pos[head_word_pos-1]

                    heads_here.append(head_token_pos)
                    deprel_ids_here.append(deprel_to_id[deprels[word_idx]])

                    tokens_representing_word_here.append(sentence_position)  # first word is index 1; index 0 is BOS

            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                heads_here.append(skip_index)
                deprel_ids_here.append(skip_index)

            previous_word_idx = word_idx

        remapped_heads.append(heads_here)
        deprel_ids.append(deprel_ids_here)
        tokens_representing_words.append(tokens_representing_word_here)
        
        num_words.append(len(tokens_representing_word_here))
        if len(tokens_representing_word_here) > maxlen_t2w:
            maxlen_t2w = len(tokens_representing_word_here)

    # pad t2w lists to same length
    for t2w in tokens_representing_words:
        t2w += [-1] * (maxlen_t2w-len(t2w))

    tokenized_inputs["head"] = remapped_heads
    tokenized_inputs["deprel_ids"] = deprel_ids
    tokenized_inputs["tokens_representing_words"] = tokens_representing_words
    tokenized_inputs["num_words"] = num_words
    tokenized_inputs["tokenid_to_wordid"] = [tokenized_inputs.word_ids(batch_index=i) for i in range(len(examples_heads))] # map token ID to word ID

    return tokenized_inputs
