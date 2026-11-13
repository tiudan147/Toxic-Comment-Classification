import pandas as pd

import re
import unicodedata
import json

from difflib import get_close_matches
from underthesea import word_tokenize
from underthesea.corpus import viet_dict_74K

raw_words = viet_dict_74K.words
vn_vocab = {w.lower() for w in raw_words} | {sub.lower() for w in raw_words for sub in w.split()}

def process_text(text):
    # đưa về chuẩn NFC
    text  = unicodedata.normalize("NFC", text)

    # làm sạch
    text = re.sub(r"<[^>]+>", " ", text) # bỏ html
    text = re.sub(r"https?://\S+|www\.\S+", " ", text) # bỏ link web
    text = re.sub(r"\d+", "", text) # bỏ số
    text = re.sub(r"\s+", " ", text).strip() # bỏ khoảng trắng thừa


    # xử lý dấu hỏi liên tiếp vd: ?????? -> ? questionrepeat
    text = re.sub(r"\?{3,}", " ? questionrepeat ", text)
    text = re.sub(r"\s*\?\s*", " ? ", text)

    # xử lý kí tự lặp vd: nguuuu -> repeat ngu
    text = re.sub(
        r"(?u)\b\w*(\w)\1+\w*\b",  
        lambda m: f"repeat  {re.sub(r'(?u)(\w)\1+', r'\1', m.group(0))}",
        text
    )

    # xử lý chữ viết hoa vd: NGU -> caps NGU
    caps = r"[A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ]"
    pattern = rf"\b({caps}{{2,}})\b"
    text = re.sub(pattern, r"caps \1", text) 

    # lập danh sách các từ viết tắt toxic và thay thế
    with open(r'dictionaries\teencode_toxic.json', 'r', encoding='utf-8') as f:
        d = {k.lower(): v for k, v in json.load(f).items()}
    pattern = re.compile(r'(?<!\w)(' + '|'.join(map(re.escape, sorted(d, key=len, reverse=True))) + r')(?!\w)', re.I)
    text = pattern.sub(lambda m: d[m.group(0).lower()], text)

    # lập danh sách teencode và thay thế
    with open(r'dictionaries\teencode.json', 'r', encoding='utf-8') as f:
        d = {k.lower(): v for k, v in json.load(f).items()}
    pattern = re.compile(r'(?<!\w)(' + '|'.join(map(re.escape, sorted(d, key=len, reverse=True))) + r')(?!\w)', re.I)
    text = pattern.sub(lambda m: d[m.group(0).lower()], text)

    # giữ lại các kí tự có nghĩa
    pattern = (
            r"[^"
            r"a-zA-Z0-9\s.,?"
            r"àáảãạăằắẳẵặâầấẩẫậ"
            r"èéẻẽẹêềếểễệ"
            r"ìíỉĩị"
            r"òóỏõọôồốổỗộơờớởỡợ"
            r"ùúủũụưừứửữự"
            r"ỳýỷỹỵđ"
            r"ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ"
            r"ÈÉẺẼẸÊỀẾỂỄỆ"
            r"ÌÍỈĨỊ"
            r"ÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ"
            r"ÙÚỦŨỤƯỪỨỬỮỰ"
            r"ỲÝỶỸỴĐ"
            r"]+"
        )
    text = re.sub(pattern, "", text)


    # sửa chính tả, chuẩn hóa từ viết tắt bằng khoảng cách Levenshtein
    fixed_words = [
        t if (t.lower() in vn_vocab or len(t) < 3 or not t.isalpha()) # từ ngắn quá ko có trong từ điển sẽ bị ép sang từ sai nghĩa, những kí tự .,? cũng bỏ qua
        else (get_close_matches(t.lower(), vn_vocab, n=1, cutoff=0.8) or [t])[0]
        for t in word_tokenize(text)
    ]
    text = word_tokenize(" ".join(fixed_words), format="text")

    special_tokens = ["repeat", "caps", "questionrepeat"]
    for tok in special_tokens:
        text = re.sub(rf"\b{tok}_", f"{tok} ", text)   # gỡ nối phía sau: repeat_ngu -> repeat ngu
        text = re.sub(rf"_{tok}\b", f" {tok}", text)   # gỡ nối phía trước nếu có: abc_repeat -> abc repeat

    # chuyển toàn bộ về chữ thường
    text = re.sub(r"\s+", " ", text).strip() # bỏ khoảng trắng thừa
    text = text.lower()

    return text

if __name__ == "__main__":
    text = "Địt mẹ mày làm sao ???????"
    print(process_text(text))
