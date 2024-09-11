import numpy as np
from typing import Optional


def get_word_character_positions(string):
    """
    Takes a string and returns a list of tuples, each containing the start and end character positions of a word in the string.
    :param string (str):
        the input string
    """
    words = string.split()
    char_index_pairs = []
    pos = 0
    for w in words:
        char_index_pairs.append((pos, pos + len(w)))
        pos += len(w) + 1
    return char_index_pairs


def aggregate_score_by_word(string, scores, offsets, mode, tokenizer=None):
    # todo: do not aggregate punctuation when attached to word
    try:
        scores = [s.item() for s in scores]
    except AttributeError:
        pass

    agg_scores = []
    word_mapping = get_word_character_positions(string)
    current_word_index = 0
    if mode == 'first':
        for score, index in zip(scores, offsets):
            start, end = index
            start_current_word, end_current_word = word_mapping[current_word_index]
            if (start + 1 == start_current_word) or (start == start_current_word == 0):
                agg_scores.append(score)
            if end == end_current_word:
                current_word_index += 1
    elif mode == 'sum':
        current_score = 0
        for score, index in zip(scores, offsets):
            current_score += score
            _, end = index
            _, end_current_word = word_mapping[current_word_index]
            if end == end_current_word:
                agg_scores.append(current_score)
                current_score = 0
                current_word_index += 1
    elif mode == 'multiply':
        current_score = 1
        for score, index in zip(scores, offsets):
            current_score *= score
            _, end = index
            _, end_current_word = word_mapping[current_word_index]
            if end == end_current_word:
                agg_scores.append(current_score)
                current_score = 1
                current_word_index += 1
    elif mode == 'mean':
        current_score, current_token_count = 0, 0
        for score, index in zip(scores, offsets):
            current_score += score
            current_token_count += 1
            _, end = index
            _, end_current_word = word_mapping[current_word_index]
            if end == end_current_word:
                agg_scores.append(current_score / current_token_count)
                current_score, current_token_count = 0, 0
                current_word_index += 1
    elif mode == 'string':
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided for mode 'string'")
        current_score = []
        for score, index in zip(scores, offsets):
            current_score.append(score)
            start, end = index
            start_current_word, end_current_word = word_mapping[current_word_index]
            if end == end_current_word:
                agg_scores.append(tokenizer.decode(current_score).strip())
                current_score = []
                current_word_index += 1
    return agg_scores


# def map_words_to_token_positions(string, tokenizer_offset_mapping):
#     """
#     Map words in a string to token positions in the tokenized string.

#     :param string (str):
#         the input string
#     :param tokenizer (transformers.PreTrainedTokenizer):
#         the tokenizer used to tokenize the string
#     :param tokenizer_offset_mapping (list):
#         the offset mapping of the tokenized string
#     :return (list):
#         a list of token positions
#     """

#     # get character positions of words in string
#     word_character_positions = get_word_character_positions(string)[1:]

#     token_positions = []
#     for (word_start, word_end) in word_character_positions:
#         word_pos = []
#         for i, (token_start, token_end) in enumerate(tokenizer_offset_mapping):
#             if token_start >= word_start - 1 and token_end <= word_end:
#                 word_pos.append(i)
#         token_positions.append(word_pos[0])
#     # add the last token position
#     token_positions.append(len(tokenizer_offset_mapping))

#     return token_positions
