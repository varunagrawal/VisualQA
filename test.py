import json
import pickle
import h5py
from tqdm import tqdm


def compare_aid_to_ans(od, myd):
    # do they have the same number of answers?
    print("Original dataset length:", len(od))
    print("My dataset length:", len(myd))
    
    # ok so let's compare answers
    mismatch = False
    for i in range(len(od)):
        if od[str(i+1)] != myd[i]:  # od is 1-indexed to work with Lua
            print(i, od[str(i)], myd[i])
            mismatch = True

    if not mismatch:
        print("Yay all the answers match")


def compare_wid_to_word(od, myd):
    print(len(od))
    print(len(myd))

    # ok so let's compare words
    mismatch = False
    for i in range(len(od)):
        if myd[i+1] not in od.values(): 
            print(i, myd[i+1])
            mismatch = True

    if not mismatch:
        print("Yay all the words match")
    else:
        print("Oops. Some words don't match")


def compare_questions(od_questions, od_ques_id, od_ques_len, od_id_to_word, od_ans, myd):
    print(len(od_questions))
    print(len(myd))
    
    dataset = {}
    for x in myd:
        dataset[x['question_id']] = x

    for i in tqdm(range(len(od_questions)), total=len(od_questions)):
        qid = od_ques_id[i]
        q = dataset[qid]
        assert od_questions[i].shape[0] == q['question_wids'].shape[0], i
        od_q = ' '.join([od_id_to_word[str(t)] for t in od_questions[i] if t != 0])
        myd_q = ' '.join(q['question_tokens_UNK'])
        print(od_q, myd_q)
        assert od_q == myd_q
        assert od_ques_len[i] == q['question_length'], i 
        assert od_ans[i] == q['answer_id'] + 1, "{0}, {1}, {2}".format(i, od_ans[i], q['answer_id'])

    print("Done comparing questions")


# this is `od`
orig_d = json.load(open('/home/vagrawal/projects/VQA_LSTM_CNN/data_prepro.json'))
f = h5py.File('/home/vagrawal/projects/VQA_LSTM_CNN/data_prepro.h5', 'r')
ques_train = f['ques_train']
ques_length_train = f['ques_length_train']
answers = f['answers']
ques_id_train = f['question_id_train']

dataset, vocab, word_to_wid, wid_to_word, ans_to_aid, aid_to_ans = pickle.load(open('vqa_train_dataset_cache.pickle', 'rb'))


compare_aid_to_ans(orig_d['ix_to_ans'], aid_to_ans)
compare_wid_to_word(orig_d['ix_to_word'], wid_to_word)
compare_questions(ques_train, ques_id_train, ques_length_train, orig_d['ix_to_word'], answers, dataset)