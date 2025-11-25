import logging
import os
import pickle
# utils.py
import re
import time

import torch

# utils_phobert_patch.py
# -*- coding: utf-8 -*-
"""
Helpers for PhoBERT: Vietnamese word segmentation and char->token span mapping.
Includes a fallback to build offset_mapping when using slow (Python) tokenizers.
"""

# utils_phobert_patch.py
import unicodedata
from typing import Any, Dict, List, Optional, Tuple


def normalize_vi(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00A0", " ")
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def raw_to_segment_span(raw_sent: str, seg_sent: str,
                        raw_start: int, raw_end: int) -> Optional[Tuple[int,int]]:
    """
    raw_* là vị trí ký tự trên câu gốc (không có dấu '_'),
    map sang span trên câu đã segment (có '_' nối âm tiết).
    Cho phép '_' hoặc ' ' thay nhau giữa các âm tiết.
    """
    sub = raw_sent[raw_start:raw_end]
    if not sub:
        return None
    pattern = re.escape(sub)
    # khoảng trắng trong raw có thể thành '_' trong seg
    pattern = re.sub(r'\\\s+', r'[_ ]+', pattern)
    m = re.search(pattern, seg_sent)
    if not m:
        return None
    return (m.start(), m.end())

def vn_word_segment(text: str) -> str:
    """
    Segment TV và nối âm tiết bằng '_' để hợp vocab PhoBERT.
    Yêu cầu: pip install underthesea
    """
    try:
        from underthesea import word_tokenize  # type: ignore
        return word_tokenize(text, format="text")  # ví dụ: "Hà_Nội là thủ_đô ..."
    except Exception:
        # Fallback nếu thiếu thư viện: trả nguyên văn (KHÔNG tối ưu với PhoBERT)
        return text

def char_to_token_span(offset_mapping: List[Tuple[int, int]],
                       start_char: int,
                       end_char: int) -> Optional[Tuple[int,int]]:
    """
    Map khoảng ký tự [start_char, end_char) trong câu đã segment
    sang (tok_start, tok_end) (inclusive) theo offset_mapping.
    Bỏ qua special tokens có offset (0,0).
    """
    tok_s = tok_e = None
    for i, (s, e) in enumerate(offset_mapping):
        if s == 0 and e == 0:
            continue
        if tok_s is None and not (e <= start_char or s >= end_char):
            tok_s = i
        if tok_s is not None and not (e <= start_char or s >= end_char):
            tok_e = i
    return None if tok_s is None or tok_e is None else (tok_s, tok_e)


def _slow_build_offsets_from_tokens(seg_sent: str, tokens: List[str]) -> List[Tuple[int, int]]:
    """
    Xây offset_mapping cho tokenizer chậm (Python). Giả định token kiểu SentencePiece,
    với tiền tố '▁' đánh dấu bắt đầu từ (tương ứng một khoảng trắng trong chuỗi).
    Trả về danh sách (start, end) theo seg_sent, gồm cả special tokens như (0,0).
    """
    # Chuẩn hoá chuỗi (không strip) vì seg_sent đã là output segmenter với khoảng trắng giữa từ.
    i = 0
    offsets: List[Tuple[int, int]] = []
    for tok in tokens:
        # Special tokens thường là '<s>', '</s>', '<pad>'...
        if tok.startswith('<') and tok.endswith('>'):
            offsets.append((0, 0))
            continue
        piece = tok
        # SentencePiece dùng '▁' để biểu diễn khoảng trắng trước từ
        if piece.startswith('▁'):
            piece = piece[1:]
            # Bỏ qua các space thừa nếu có
            while i < len(seg_sent) and seg_sent[i] == ' ':
                i += 1
        # Nếu token rỗng (có thể hiếm), map vào (i,i)
        if len(piece) == 0:
            offsets.append((i, i))
            continue
        start = i
        end = start + len(piece)
        # Trường hợp hiếm khi ký tự không trùng (do normal hoá), ta sẽ tìm khớp gần nhất.
        if seg_sent[start:end] != piece:
            # thử dịch sang phải đến khi khớp (an toàn, nhưng giữ bounded)
            j = start
            found = False
            while j <= min(len(seg_sent) - len(piece), start + 10):  # không trôi quá xa
                if seg_sent[j:j+len(piece)] == piece:
                    start = j
                    end = j + len(piece)
                    found = True
                    break
                j += 1
            # nếu vẫn không khớp, chấp nhận best-effort
        offsets.append((start, end))
        i = end
    return offsets


def encode_with_offsets(tokenizer, seg_sent: str, max_length: int = 512) -> Dict[str, Any]:
    """
    Gọi tokenizer để lấy input_ids/attention_mask và luôn trả về offset_mapping.
    - Nếu tokenizer là fast: dùng return_offsets_mapping=True.
    - Nếu tokenizer là slow: tự dựng offset_mapping từ token chuỗi.
    Trả về dict có các khoá: input_ids, attention_mask, offset_mapping.
    """
    is_fast = getattr(tokenizer, "is_fast", False)
    if is_fast:
        enc = tokenizer(
            seg_sent,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "offset_mapping": enc["offset_mapping"],
        }
    else:
        # Slow path
        enc = tokenizer.encode_plus(
            seg_sent,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True
        )
        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        offsets = _slow_build_offsets_from_tokens(seg_sent, tokens)
        # Đảm bảo độ dài tương ứng
        if len(offsets) != len(input_ids):
            # fallback an toàn: pad/cắt cho khớp
            if len(offsets) < len(input_ids):
                offsets += [(0,0)] * (len(input_ids) - len(offsets))
            else:
                offsets = offsets[:len(input_ids)]
        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "offset_mapping": offsets,
        }


def get_device(prefer=None):
    """Return a torch.device based on prefer ('cuda'|'mps'|'cpu') or environment.
    Handles Apple MPS fallback and ensures a valid device is returned."""
    if prefer is None:
        prefer = os.environ.get('DEVICE', None)
    if prefer == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    if prefer == 'mps':
        # MPS support (Apple Silicon)
        try:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                return torch.device('mps')
        except Exception:
            pass
    # auto-detect
    if torch.cuda.is_available():
        return torch.device('cuda')
    try:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device('mps')
    except Exception:
        pass
    return torch.device('cpu')


def move_to_device(obj, device):
    """Move tensor or module or nested structure to device."""
    if hasattr(obj, 'to') and not isinstance(obj, (str, bytes)):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [move_to_device(v, device) for v in obj]
        return type(obj)(t)
    return obj


def get_logger(dataset):
    pathname = "./log/{}_{}.txt".format(dataset, time.strftime("%m-%d_%H-%M-%S"))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(pathname)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def save_file(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_file(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def convert_index_to_text(index, type):
    text = "-".join([str(i) for i in index])
    text = text + "-#-{}".format(type)
    return text


def decode(outputs, labels, tri_args):
    results = {}
    arg_dict = {}

    for key in ["ti", "tc", "ai", "ac"]:
        pred = outputs[key]
        if pred is None:
            pred_set = set()
        else:
            pred_set = set([tuple(x.tolist()) for x in pred])
        if key == "ai":
            for b, x, y, e in pred_set:
                for c in range(x, y + 1):
                    if (b, c) in arg_dict:
                        arg_dict[(b, c)].append((b, x, y))
                    else:
                        arg_dict[(b, c)] = [(b, x, y)]
        if key in ["ac"]:
            new_pred_set = set()
            for b, x, e, r in pred_set:
                if (b, x) in arg_dict:
                    for prefix in arg_dict[(b, x)]:
                        new_pred_set.add(prefix + (e, r))
            pred_set = set([x for x in new_pred_set if (x[-2], x[-1]) in tri_args])
        results[key + "_r"] = len(labels[key])
        results[key + "_p"] = len(pred_set)
        results[key + "_c"] = len(pred_set & labels[key])

    return results

def calculate_f1(r, p, c):
    if r == 0 or p == 0 or c == 0:
        return 0, 0, 0
    r = c / r
    p = c / p
    f1 = (2 * r * p) / (r + p)
    return f1, r, p
