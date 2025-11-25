# utils_phobert_patch.py
# -*- coding: utf-8 -*-
"""
Helpers for PhoBERT: Vietnamese word segmentation and char->token span mapping.
Includes a fallback to build offset_mapping when using slow (Python) tokenizers.
"""

from typing import Optional, Tuple, List, Dict, Any

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
