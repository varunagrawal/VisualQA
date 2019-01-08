"""
Utils to help in text processing

Author: Vaurn Agrawal (varunagrawal)
"""

import re
from tqdm import tqdm
import numpy as np
import nltk


def tokenize(sentence):
    sentence = sentence.lower()
    return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) if
            i != '' and i != ' ' and i != '\n']


def preprocess_questions(dataset, method="nltk", display=True):
    for idx, d in enumerate(tqdm(dataset, leave=display)):
        s = d["question"]
        if method == "nltk":
            d["question_tokens"] = nltk.word_tokenize(str(s).lower())
        else:
            d["question_tokens"] = tokenize(s)
    return dataset


def get_vocabulary(dataset, min_word_count=0):
    """
    Filter out words in the questions that are <= min_word_count and create a vocabulary from the filtered words
    :param dataset: The VQA dataset
    :param min_word_count: The minimum number of counts the word needs in order to be included
    :return:
    """
    counts = {}
    print("Calculating word counts in questions")
    for d in dataset:
        for w in d["question_tokens"]:
            counts[w] = counts.get(w, 0) + 1

    vocab = [w for w, n in counts.items() if n > min_word_count]

    # cw = sorted([(n, w) for w, n in counts.items() if n > min_word_count], reverse=True)
    # print('\n'.join(map(str, cw[:20])))

    # Add the 'UNK' token
    vocab.append('UNK')  # UNK has it's own ID

    return vocab


def remove_tail_words(dataset, vocab, display=True):
    if display:
        print("Removing tail words")

    for idx, d in enumerate(tqdm(dataset, leave=display)):
        words = d["question_tokens"]
        question = [w if w in vocab else 'UNK' for w in words]
        d["question_tokens"] = question

    return dataset


def encode_questions(dataset, word_to_wid, max_length=25, display=True):
    """
    Encode each question into a vector of size Max_Length x Vocab_Size
    :param dataset:
    :param word_to_wid:
    :param max_length
    :param display
    :return:
    """
    if display:
        print("Encoding the questions")

    for idx, d in enumerate(tqdm(dataset, leave=display)):
        d["question_length"] = min(len(d["question_tokens"]), max_length)
        d["question_wids"] = np.zeros(max_length, dtype=np.int32)  # 0 -> UNK

        for k, w in enumerate(d["question_tokens"]):
            if k < max_length:
                wid = word_to_wid.get(w, word_to_wid["UNK"])
                # ensure it is an int so it can be used for indexing
                d["question_wids"][k] = int(wid)

    return dataset


def get_top_answers(dataset, top=1000, display=True):
    print("Finding top {0} answers".format(top))
    counts = {}
    for idx, d in enumerate(tqdm(dataset, leave=display)):
        ans = d["answer"].lower()
        counts[ans] = counts.get(ans, 0) + 1

    print("{0} unqiue answers".format(len(counts)))

    # Get a list of answers sorted by how common they are
    ans_counts = sorted([(count, ans)
                         for ans, count in counts.items()], reverse=True)
    top_answers = []

    for i in range(top):
        top_answers.append(ans_counts[i][1])

    if display:
        print("The top 10 answers are:")
        print("\n".join(map(str, ans_counts[:10])))

    return top_answers


def encode_answers(dataset, ans_to_aid, display=True):
    print("Encoding answers")
    for d in tqdm(dataset, leave=display):
        d["answer_id"] = ans_to_aid[d['answer'].lower()]

    return dataset


def filter_dataset(dataset, top_answers, display=True):
    filtered_dataset = []
    for d in tqdm(dataset, leave=display):
        if d["answer"] in top_answers:
            filtered_dataset.append(d)

    print("Original Dataset Size: ", len(dataset))
    print("Filtered Dataset Size: ", len(filtered_dataset))
    return filtered_dataset


def process_single_question(question, vocab, word_to_wid, max_length=25):
    d = [{"question": question}]
    d = preprocess_questions(d, display=False)
    d = remove_tail_words(d, vocab, display=False)
    encoded_question = encode_questions(d, word_to_wid,
                                        max_length, display=False)
    return encoded_question[0]
