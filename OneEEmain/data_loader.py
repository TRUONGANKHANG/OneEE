# data_loader_phobert_patch.py
# -*- coding: utf-8 -*-

import json
import os
import random
from collections import defaultdict

import numpy as np
import prettytable as pt
import torch
import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from utils import (char_to_token_span, encode_with_offsets, normalize_vi,
                   raw_to_segment_span, vn_word_segment)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

SAMPLE_NUM = 0
DEBUG_SAMPLE_RATE = 0.01  # in ra ~1% câu để kiểm tra mapping; chỉnh thấp/higher tuỳ ý

class Vocabulary(object):
    PAD = "<pad>"
    UNK = "<unk>"
    ARG = "<arg>"

    def __init__(self):
        self.tri_label2id = {}  # trigger
        self.tri_id2label = {}
        self.tri_id2count = defaultdict(int)
        self.tri_id2prob = {}

        self.rol_label2id = {}  # role
        self.rol_id2label = {}

    def label2id(self, label, type):
        label = label.lower()
        if type == "tri":
            return self.tri_label2id[label]
        elif type == "rol":
            return self.rol_label2id[label]
        else:
            raise Exception("Wrong Label Type!")

    def add_label(self, label, type):
        label = label.lower()

        if type == "tri":
            if label not in self.tri_label2id:
                self.tri_label2id[label] = len(self.tri_id2label)
                self.tri_id2label[self.tri_label2id[label]] = label
                self.tri_id2count[self.tri_label2id[label]] += 1
        elif type == "rol":
            if label not in self.rol_label2id:
                self.rol_label2id[label] = len(self.rol_id2label)
                self.rol_id2label[self.rol_label2id[label]] = label
        else:
            raise Exception("Wrong Label Type!")

    def get_prob(self):
        total = np.sum(list(self.tri_id2count.values()))
        for k, v in self.tri_id2count.items():
            self.tri_id2prob[k] = v / total if total > 0 else 0.0

    @property
    def tri_label_num(self):
        return len(self.tri_label2id)

    @property
    def rol_label_num(self):
        return len(self.rol_label2id)

    @property
    def label_num(self):
        return self.tri_label_num


def collate_fn(data):
    inputs, att_mask, word_mask1d, word_mask2d, triu_mask2d, tri_labels, arg_labels, role_labels, tuple_labels, event_list, training = map(
        list, zip(*data))

    batch_size = len(inputs)
    max_tokens = np.max([x.shape[0] for x in word_mask1d])

    inputs = pad_sequence(inputs, True)
    att_mask = pad_sequence(att_mask, True)
    word_mask1d = pad_sequence(word_mask1d, True)

    def pad_2d(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = x
        return new_data

    def pad_3d(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :, :x.shape[1], :x.shape[2]] = x
        return new_data

    def pad_4d(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :, :x.shape[1], :x.shape[2], :] = x
        return new_data
    word_mat = torch.zeros((batch_size, max_tokens, max_tokens), dtype=torch.bool)
    word_mask2d = pad_2d(word_mask2d, word_mat)
    triu_mat = torch.zeros((batch_size, max_tokens, max_tokens), dtype=torch.bool)
    triu_mask2d = pad_2d(triu_mask2d, triu_mat)
    tri_mat = torch.zeros((batch_size, tri_labels[0].size(0), max_tokens, max_tokens), dtype=torch.bool)
    tri_labels = pad_3d(tri_labels, tri_mat)
    arg_mat = torch.zeros((batch_size, arg_labels[0].size(0), max_tokens, max_tokens), dtype=torch.bool)
    arg_labels = pad_3d(arg_labels, arg_mat)
    role_mat = torch.zeros((batch_size, role_labels[0].size(0), max_tokens, max_tokens, role_labels[0].size(-1)), dtype=torch.bool)
    role_labels = pad_4d(role_labels, role_mat)

    _tuple_labels = {k: set() for k in ["ti", "tc", "ai", "ac"]}
    if not training[0]:
        for i, x in enumerate(tuple_labels):
            for k, v in x.items():
                _tuple_labels[k] = _tuple_labels[k] | set([(i,) + t for t in x[k]])
        role_label_num = len(_tuple_labels["ac"])
    else:
        role_label_num = np.sum([len(x["ac"]) for x in tuple_labels])

    event_idx = []
    for b in range(inputs.size(0)):
        pos_event, neg_events = event_list[b]
        neg_list = random.choices(neg_events, k=SAMPLE_NUM) if len(neg_events) > 0 else []
        event_idx.append([pos_event] + neg_list)
    event_idx = torch.LongTensor(event_idx)
    return inputs, att_mask, word_mask1d, word_mask2d, triu_mask2d, tri_labels, arg_labels, role_labels, event_idx, _tuple_labels, role_label_num


class RelationDataset(Dataset):
    def __init__(self, inputs, att_mask, word_mask1d, word_mask2d, triu_mask2d, tri_labels, arg_labels,
                 role_labels, gold_tuples, event_list):
        self.inputs = inputs
        self.att_mask = att_mask
        self.word_mask1d = word_mask1d
        self.word_mask2d = word_mask2d
        self.triu_mask2d = triu_mask2d
        self.tri_labels = tri_labels
        self.arg_labels = arg_labels
        self.role_labels = role_labels
        self.tuple_labels = gold_tuples
        self.event_list = event_list
        self.training = True

    def eval_data(self):
        self.training = False

    def __getitem__(self, item):
        return torch.LongTensor(self.inputs[item]), \
               torch.LongTensor(self.att_mask[item]), \
               torch.BoolTensor(self.word_mask1d[item]), \
               torch.BoolTensor(self.word_mask2d[item]), \
               torch.BoolTensor(self.triu_mask2d[item]), \
               torch.BoolTensor(self.tri_labels[item]), \
               torch.BoolTensor(self.arg_labels[item]), \
               torch.BoolTensor(self.role_labels[item]), \
               self.tuple_labels[item], \
               self.event_list[item], \
               self.training

    def __len__(self):
        return len(self.inputs)


def process_bert(data, tokenizer, vocab):
    inputs = []
    att_mask = []
    word_mask1d = []
    word_mask2d = []
    triu_mask2d = []
    arg_labels = []
    tri_labels = []
    role_labels = []
    gold_tuples = []
    event_list = []

    total_event_set = set([i for i in range(vocab.tri_label_num)])

    for ins_id, instance in tqdm.tqdm(enumerate(data), total=len(data)):

        tokens_list = instance.get("content", instance.get("tokens", []))
        if (not tokens_list) and ("sentence" in instance):
            tokens_list = instance["sentence"].split()

        # 1) Chuẩn hoá + segment -> tokenize
        raw_sent = " ".join(tokens_list)
        raw_sent = normalize_vi(raw_sent)
        seg_sent = vn_word_segment(raw_sent)

        enc = encode_with_offsets(
            tokenizer,
            seg_sent,
            max_length=getattr(tokenizer, "model_max_length", 512)
        )
        _inputs = enc["input_ids"]
        _att_mask = np.array(enc["attention_mask"])
        _offset = enc["offset_mapping"]

        # 2) Chiều dài phần nội dung (bỏ <s> và </s>)
        length = int(_att_mask.sum()) - 2
        if length <= 0:
            continue

        # _word_mask1d = np.array([1] * length)
        # _word_mask2d = np.triu(np.ones((length, length), dtype=bool))
        # # _triu_mask2d = np.ones((length, length), dtype=bool)trainer
        # _triu_mask2d = np.triu(np.ones((length, length), dtype=bool), k=1)
        # np.fill_diagonal(_triu_mask2d, 0)
        _word_mask1d = np.ones(length, dtype=np.int64)

        # Cho trigger/argument: cho phép mọi cặp (i,j) trong câu
        _word_mask2d = np.ones((length, length), dtype=bool)

        # Cho role: chỉ giữ tam giác trên, bỏ đường chéo
        _triu_mask2d = np.triu(np.ones((length, length), dtype=bool), k=1)
        np.fill_diagonal(_triu_mask2d, 0)


        _tri_labels = np.zeros((vocab.tri_label_num, length, length), dtype=bool)
        _arg_labels = np.zeros((vocab.tri_label_num, length, length), dtype=bool)
        _role_labels = np.zeros((vocab.tri_label_num, length, length, vocab.rol_label_num), dtype=bool)

        # ---- entity index cho arguments (entity_id -> RAW char spans) ----
        ent_map = {}
        for ent in instance.get("entity_mentions", []):
            ent_map[ent["id"]] = (ent.get("start_char"), ent.get("end_char"))

        # ---- events ----
        if "event_type" in instance:
            pos_event = vocab.label2id(instance["event_type"], "tri")
        else:
            pos_event = 0

        event_set = set()
        _gold_tuples = {k: set() for k in ["ti", "tc", "ai", "ac"]}
        events = instance.get("events", instance.get("event_mentions", []))

        assigned = 0  # debug counter

        for event in events:
            trig = event.get("trigger", {})
            # Map RAW -> SEG -> TOKEN
            t_sc_raw, t_ec_raw = trig.get("start_char"), trig.get("end_char")
            seg_span = raw_to_segment_span(raw_sent, seg_sent, t_sc_raw, t_ec_raw)
            if not seg_span:
                continue
            t_sc_seg, t_ec_seg = seg_span

            t_tok_span = char_to_token_span(_offset, t_sc_seg, t_ec_seg)
            if not t_tok_span:
                continue
            t_s_tok, t_e_tok = t_tok_span

            # Bỏ <s>: chuyển sang hệ tọa độ nội dung [0..length-1]
            t_s = max(0, min(t_s_tok - 1, length - 1))
            t_e = max(0, min(t_e_tok - 1, length - 1))
            if t_s > t_e:
                continue

            event_type = vocab.label2id(event.get("type", event.get("event_type", "UNK")), "tri")

            _tri_labels[event_type, t_s, t_e] = 1
            _gold_tuples["ti"].add((t_s, t_e))
            _gold_tuples["tc"].add((t_s, t_e, event_type))
            event_set.add(event_type)
            assigned += 1

            # Arguments qua entity_id
            for arg in event.get("args", event.get("arguments", [])):
                ent_id = arg.get("entity_id")
                if (not ent_id) or (ent_id not in ent_map):
                    continue
                a_sc_raw, a_ec_raw = ent_map[ent_id]
                seg_span = raw_to_segment_span(raw_sent, seg_sent, a_sc_raw, a_ec_raw)
                if not seg_span:
                    continue
                a_sc_seg, a_ec_seg = seg_span

                a_tok_span = char_to_token_span(_offset, a_sc_seg, a_ec_seg)
                if not a_tok_span:
                    continue
                a_s_tok, a_e_tok = a_tok_span

                a_s = max(0, min(a_s_tok - 1, length - 1))
                a_e = max(0, min(a_e_tok - 1, length - 1))
                if a_s > a_e:
                    continue

                role_name = arg.get("role", "UNK")
                role = vocab.label2id(role_name, "rol")
                _arg_labels[event_type, a_s, a_e] = 1
                _role_labels[event_type, t_s:t_e+1, a_s:a_e+1, role] = 1
                _gold_tuples["ai"].add((a_s, a_e, event_type))
                _gold_tuples["ac"].add((a_s, a_e, event_type, role))
                assigned += 1

        neg_event = list(total_event_set - event_set)

        # Debug UNK
        unk_id = tokenizer.unk_token_id
        if unk_id is not None:
            unk_ratio = sum(1 for x in _inputs if x == unk_id) / max(1, len(_inputs))
            if unk_ratio > 0.10:
                print(f"[WARN] PhoBERT UNK ratio high: {unk_ratio:.2%} (ins_id={ins_id})")

        # ----- Debug: in ra một số câu để kiểm tra mapping -----
        # if len(events) > 0 and random.random() < DEBUG_SAMPLE_RATE:
        #     print("\n[DEBUG] ========= Example:", instance.get("sent_id", ins_id))
        #     print("RAW  :", raw_sent)
        #     print("SEG  :", seg_sent)
            # for ev in events:
            #     trig = ev.get("trigger", {})
            #     print("  > Event:", ev.get("event_type", ev.get("type")))
            #     print("    Trigger text:", trig.get("text"), "| raw-char:", trig.get("start_char"), trig.get("end_char"))
            #     seg_span = raw_to_segment_span(raw_sent, seg_sent, trig.get("start_char"), trig.get("end_char"))
            #     print("    Mapped seg-span:", seg_span)
            #     if seg_span:
            #         seg_sub = seg_sent[seg_span[0]:seg_span[1]]
            #         print("    SEG substring:", seg_sub)
            #     for arg in ev.get("arguments", ev.get("args", [])):
            #         ent_id = arg.get("entity_id")
            #         if ent_id in ent_map:
            #             a_sc, a_ec = ent_map[ent_id]
            #             seg_span = raw_to_segment_span(raw_sent, seg_sent, a_sc, a_ec)
            #             print(f"      Arg({arg.get('role')}):", arg.get("text"), "| raw-span:", (a_sc, a_ec), "-> seg-span:", seg_span)
            #             if seg_span:
            #                 print("         SEG substring:", seg_sent[seg_span[0]:seg_span[1]])
            # if assigned == 0:
            #     print("  [WARN] No labels assigned for this instance.")

        inputs.append(_inputs)
        att_mask.append(_att_mask)
        word_mask1d.append(_word_mask1d)
        word_mask2d.append(_word_mask2d)
        triu_mask2d.append(_triu_mask2d)
        arg_labels.append(_arg_labels)
        tri_labels.append(_tri_labels)
        role_labels.append(_role_labels)
        gold_tuples.append(_gold_tuples)
        event_list.append((pos_event, neg_event))
        # === STAT CHECK mỗi 2000 mẫu ===
        # if len(inputs) % 2000 == 0:
        #     tri_pos = sum(x.sum() for x in tri_labels[-2000:])
        #     arg_pos = sum(x.sum() for x in arg_labels[-2000:])
        #     role_pos = sum(x.sum() for x in role_labels[-2000:])
        #     print(f"[STAT] last2k pos -> tri:{tri_pos} arg:{arg_pos} role:{role_pos}")

    return inputs, att_mask, word_mask1d, word_mask2d, triu_mask2d, tri_labels, arg_labels, role_labels, gold_tuples, event_list


def fill_vocab(vocab, dataset):
    statistic = {"tri_num": 0, "arg_num": 0}
    for instance in dataset:
        events = instance.get("events", instance.get("event_mentions", []))
        for eve in events:
            vocab.add_label(eve.get("type", eve.get("event_type", "None")), "tri")
            args = eve.get("args", eve.get("arguments", []))
            for arg in args:
                role = arg.get("role", "None")
                vocab.add_label(role, "rol")
            statistic["arg_num"] += len(args)
        statistic["tri_num"] += len(events)
    return statistic


def load_data(config):
    global SAMPLE_NUM
    with open("./data/{}/train.json".format(config.dataset), "r", encoding="utf-8") as f:
        train_data = [json.loads(x) for x in f.readlines()]
    with open("./data/{}/dev.json".format(config.dataset), "r", encoding="utf-8") as f:
        dev_data = [json.loads(x) for x in f.readlines()]
    with open("./data/{}/test.json".format(config.dataset), "r", encoding="utf-8") as f:
        test_data = [json.loads(x) for x in f.readlines()]

    tokenizer = AutoTokenizer.from_pretrained(config.bert_name, cache_dir="./cache/", use_fast=True)
    config.tokenizer = tokenizer

    vocab = Vocabulary()
    train_statistic = fill_vocab(vocab, train_data)
    vocab.get_prob()
    dev_statistic = fill_vocab(vocab, dev_data)
    test_statistic = fill_vocab(vocab, test_data)

    with open("./data/{}/ty_args.json".format(config.dataset), "r", encoding="utf-8") as f:
        tri_args = json.load(f)
    config.tri_args = set()
    for k, vs in tri_args.items():
        for v in vs:
            k_i, v_i = vocab.label2id(k, "tri"), vocab.label2id(v, "rol")
            config.tri_args.add((k_i, v_i))

    table = pt.PrettyTable([config.dataset, "#sentence", "#event", "#argument"])
    table.add_row(["train", len(train_data)] + [train_statistic[key] for key in ["tri_num", "arg_num"]])
    table.add_row(["dev", len(dev_data)] + [dev_statistic[key] for key in ["tri_num", "arg_num"]])
    table.add_row(["test", len(test_data)] + [test_statistic[key] for key in ["tri_num", "arg_num"]])
    config.logger.info("\n{}".format(table))

    config.tri_label_num = vocab.tri_label_num
    config.rol_label_num = vocab.rol_label_num
    config.label_num = vocab.tri_label_num
    config.vocab = vocab

    SAMPLE_NUM = getattr(config, "event_sample", 0)

    print("Processing train data...")
    train_dataset = RelationDataset(*process_bert(train_data, tokenizer, vocab))
    print("Processing dev data...")
    dev_dataset = RelationDataset(*process_bert(dev_data, tokenizer, vocab))
    print("Processing test data...")
    test_dataset = RelationDataset(*process_bert(test_data, tokenizer, vocab))

    dev_dataset.eval_data()
    test_dataset.eval_data()
    return train_dataset, dev_dataset, test_dataset
