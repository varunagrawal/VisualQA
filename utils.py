import re
from tqdm import tqdm
import numpy as np


def tokenize(s):
    t_str = s.lower()
    for i in [r'\?',r'\!',r'\'',r'\"',r'\$',r'\:',r'\@',r'\(',r'\)',r'\,',r'\.',r'\;']:
        t_str = re.sub( i, '', t_str)
    for i in [r'\-',r'\/']:
        t_str = re.sub( i, ' ', t_str)
    q_list = re.sub(r'\?','',t_str.lower()).split(' ')
    q_list = list(filter(lambda x: len(x) > 0, q_list))
    return q_list


def preprocess_questions(dataset):
    for idx, d in enumerate(tqdm(dataset)):
        s = d["question"]
        d["question_words"] = tokenize(s)
    return dataset


def get_vocabulary(dataset, min_word_count=1):
    """
    Filter out words in the questions that are < min_word_count and create a vocabulary from the filtered words
    :param dataset: The VQA dataset
    :param min_word_count: The minimum number of counts the word needs in order to be included
    :return:
    """
    counts = {}
    print("Calculating word counts in questions")
    for d in dataset:
        for w in d["question_words"]:
            counts[w] = counts.get(w, 0) + 1

    vocab = [w for w, n in counts.items() if n >= min_word_count]
    vocab.append('UNK')
    return vocab


def remove_tail_words(dataset, vocab):
    print("Removing tail words")
    for idx, d in enumerate(tqdm(dataset)):
        words = d["question_words"]
        question = [w if w in vocab else 'UNK' for w in words]
        d["question_words_UNK"] = question

    return dataset


def encode_questions(dataset, word_to_wid, max_length=25):
    """
    Encode each question into a vector of size Max_Length x Vocab_Size
    :param dataset:
    :param word_to_wid:
    :param max_length
    :return:
    """
    print("Encoding the questions")
    for idx, d in enumerate(tqdm(dataset)):
        d["question_length"] = min(len(d["question_words_UNK"]), max_length)
        d["question_wids"] = np.zeros(max_length, dtype=np.int32)

        for k, w in enumerate(d["question_words_UNK"]):
            if k < max_length:
                d["question_wids"][k] = int(word_to_wid[w])  # ensure it is an int so it can be used for indexing
                d['seq_length'] = len(d['question_words_UNK'])

    return dataset


def get_top_answers(dataset, top=1000):
    print("Finding top answers")
    counts = {}
    for idx, d in enumerate(tqdm(dataset)):
        ans = d["answer"]
        counts[ans] = counts.get(ans, 0) + 1

    # Get a list of answers sorted by how common they are
    ans_counts = sorted([(count, ans) for ans, count in counts.items()], reverse=True)
    top_answers = []

    for i in range(top):
        top_answers.append(ans_counts[i][1])

    return top_answers


def encode_answers(dataset, ans_to_aid):
    print("Encoding answers")
    out_of_vocab = len(ans_to_aid)

    for idx, d in enumerate(tqdm(dataset)):
        d["answer_id"] = ans_to_aid.get(d['answer'], out_of_vocab)

    return dataset


def filter_dataset(dataset, top_answers):
    filtered_dataset = []
    for idx, d in enumerate(tqdm(dataset)):
        if d["answer"] in top_answers:
            filtered_dataset.append(d)

    print("Original Dataset Size: ", len(dataset))
    print("Filtered Dataset Size: ", len(filtered_dataset))
    return filtered_dataset

