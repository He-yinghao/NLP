import json
import os
import random
import time
from collections import Counter
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from datasets import Dataset, DatasetDict
from src.models.Transformer import TransformerModel

# ---------- Data utilities ----------
SRC_LANG = "zh"  # 修改为中文
TGT_LANG = "en"

# special tokens
PAD = "<pad>"
BOS = "<sos>"
EOS = "<eos>"
UNK = "<unk>"


def tokenize_basic(text: str) -> List[str]:
    """
    对中文/英文做相应的分词：
      - 如果 src 是中文，优先尝试使用 jieba（如果已安装），否则回退到字级切分（每个汉字一个 token）
      - 英文使用按空格切分（保持原逻辑）
    这样可以避免中文整个句子被当成一个 token 的问题。
    """
    if text is None:
        return []
    text = str(text).strip()
    if SRC_LANG.lower() in ("zh", "zh-cn", "zh_tw", "chinese"):
        # try jieba if available
        try:
            import jieba  # type: ignore

            return [tok for tok in jieba.lcut(text) if tok.strip()]
        except Exception:
            # fallback: character-level tokenization
            return [ch for ch in text if not ch.isspace()]
    else:
        return text.lower().split()


def load_csv_data(
    train_path="train_zh_en.csv", test_path="test_zh_en.csv"
) -> DatasetDict:
    """从CSV文件加载数据集并添加 translation 字段。
    要求 CSV 包含列名 'zh' 和 'en'（或你指定的 SRC_LANG/TGT_LANG）。
    """
    train_df = pd.read_csv(train_path).reset_index(drop=True)
    test_df = pd.read_csv(test_path).reset_index(drop=True)

    # 将 pandas -> HuggingFace Dataset
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

    # 添加 translation 字段以兼容后续代码
    def add_translation_field(example):
        # 保证字段存在
        src_text = example.get(SRC_LANG) or example.get("zh") or ""
        tgt_text = example.get(TGT_LANG) or example.get("en") or ""
        return {"translation": {SRC_LANG: src_text, TGT_LANG: tgt_text}}

    # 对整个 DatasetDict 做 map（map 会自动对每个 split 生效）
    dataset = dataset.map(add_translation_field)
    return dataset


def build_vocabs(
    dataset: DatasetDict,
    min_freq: int = 2,
    save_dir: str = "./vocabs",
    force_rebuild: bool = False,
) -> Tuple[
    Tuple[Dict[str, int], Dict[int, str]], Tuple[Dict[str, int], Dict[int, str]]
]:
    """
    使用 dataset['train'/'valid'/'test'] 构建词表（若不存在某 split 会跳过）。
    返回 ((src_vocab, src_itos), (tgt_vocab, tgt_itos))，
    并将词表保存为 JSON 文件到 save_dir。

    参数:
        dataset: 数据集
        min_freq: 最小词频
        save_dir: 保存目录
        force_rebuild: 是否强制重新构建词表
    """

    # 检查词汇表文件是否存在
    src_vocab_path = os.path.join(save_dir, "src_vocab.json")
    tgt_vocab_path = os.path.join(save_dir, "tgt_vocab.json")

    # 如果词汇表文件存在且不强制重建，则直接读取
    if (
        not force_rebuild
        and os.path.exists(src_vocab_path)
        and os.path.exists(tgt_vocab_path)
    ):
        print("词汇表文件已存在，直接读取...")
        try:
            with open(src_vocab_path, "r", encoding="utf-8") as f:
                src_vocab = json.load(f)
            with open(
                os.path.join(save_dir, "src_itos.json"), "r", encoding="utf-8"
            ) as f:
                src_itos = json.load(f)
            with open(tgt_vocab_path, "r", encoding="utf-8") as f:
                tgt_vocab = json.load(f)
            with open(
                os.path.join(save_dir, "tgt_itos.json"), "r", encoding="utf-8"
            ) as f:
                tgt_itos = json.load(f)

            print(f"词汇表读取成功 -> 中文: {len(src_vocab)}, 英文: {len(tgt_vocab)}")
            return (src_vocab, src_itos), (tgt_vocab, tgt_itos)
        except Exception as e:
            print(f"读取词汇表文件失败: {e}，将重新构建词表...")

    # 如果文件不存在或强制重建，则执行原构建逻辑
    print("开始构建词汇表...")
    os.makedirs(save_dir, exist_ok=True)

    src_counter, tgt_counter = Counter(), Counter()

    for split in ["train", "valid", "test"]:
        if split not in dataset:
            continue
        for example in dataset[split]:
            if SRC_LANG in example and example[SRC_LANG] is not None:
                src_counter.update(tokenize_basic(example[SRC_LANG]))
            if TGT_LANG in example and example[TGT_LANG] is not None:
                tgt_counter.update(tokenize_basic(example[TGT_LANG]))

            if "translation" in example and example["translation"] is not None:
                trans = example["translation"]
                if SRC_LANG in trans and trans[SRC_LANG] is not None:
                    src_counter.update(tokenize_basic(trans[SRC_LANG]))
                if TGT_LANG in trans and trans[TGT_LANG] is not None:
                    tgt_counter.update(tokenize_basic(trans[TGT_LANG]))

    def make_vocab(counter: Counter) -> Tuple[Dict[str, int], Dict[int, str]]:
        vocab: Dict[str, int] = {PAD: 0, BOS: 1, EOS: 2, UNK: 3}
        sorted_words = sorted(
            [(w, f) for w, f in counter.items() if f >= min_freq],
            key=lambda x: (-x[1], x[0]),
        )
        idx = 4
        for w, _ in sorted_words:
            if w not in vocab:
                vocab[w] = idx
                idx += 1
        itos = {i: w for w, i in vocab.items()}
        return vocab, itos

    src_vocab, src_itos = make_vocab(src_counter)
    tgt_vocab, tgt_itos = make_vocab(tgt_counter)

    print(f"中文词表大小: {len(src_vocab)} (候选词数: {len(src_counter)})")
    print(f"英文词表大小: {len(tgt_vocab)} (候选词数: {len(tgt_counter)})")

    # 保存 JSON
    with open(os.path.join(save_dir, "src_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(src_vocab, f, ensure_ascii=False, indent=2)
    with open(os.path.join(save_dir, "src_itos.json"), "w", encoding="utf-8") as f:
        json.dump(src_itos, f, ensure_ascii=False, indent=2)

    with open(os.path.join(save_dir, "tgt_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(tgt_vocab, f, ensure_ascii=False, indent=2)
    with open(os.path.join(save_dir, "tgt_itos.json"), "w", encoding="utf-8") as f:
        json.dump(tgt_itos, f, ensure_ascii=False, indent=2)

    return (src_vocab, src_itos), (tgt_vocab, tgt_itos)


def numericalize(vocab: Dict[str, int], tokens: List[str]) -> List[int]:
    return [vocab[BOS]] + [vocab.get(t, vocab[UNK]) for t in tokens] + [vocab[EOS]]


def collate_fn(batch, src_vocab, tgt_vocab, device):
    """
    batch: list of examples (dict)，每个 example 包含 'translation':{zh:..., en:...}
    返回移动到 device 的 tensor 对
    """
    src_batch, tgt_batch = [], []
    for example in batch:
        # 可能是 example['translation'] 或直接 example['zh']/example['en']
        if "translation" in example:
            src_text = example["translation"].get(SRC_LANG, "") or ""
            tgt_text = example["translation"].get(TGT_LANG, "") or ""
        else:
            src_text = example.get(SRC_LANG, "") or ""
            tgt_text = example.get(TGT_LANG, "") or ""

        src_tok = tokenize_basic(src_text)
        tgt_tok = tokenize_basic(tgt_text)
        src_idxs = torch.tensor(numericalize(src_vocab, src_tok), dtype=torch.long)
        tgt_idxs = torch.tensor(numericalize(tgt_vocab, tgt_tok), dtype=torch.long)
        src_batch.append(src_idxs)
        tgt_batch.append(tgt_idxs)

    src_batch = pad_sequence(src_batch, padding_value=src_vocab[PAD], batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=tgt_vocab[PAD], batch_first=True)

    return src_batch.to(device), tgt_batch.to(device)


# ---------- Mask helpers ----------
def make_src_key_padding_mask(src, pad_idx):
    return src == pad_idx


def make_tgt_masks(tgt, pad_idx):
    B, T = tgt.size()
    subsequent = torch.triu(torch.ones((T, T), dtype=torch.bool), diagonal=1)
    pad_mask = tgt == pad_idx
    return subsequent.to(tgt.device), pad_mask.to(tgt.device)


# ---------- Training and evaluation ----------
def train_epoch(
    model, dataloader, optimizer, criterion, src_pad_idx, tgt_pad_idx, device, clip=1.0
):
    model.train()
    total_loss = 0.0
    for src_batch, tgt_batch in dataloader:
        tgt_input = tgt_batch[:, :-1]
        tgt_target = tgt_batch[:, 1:]
        src_key_pad_mask = make_src_key_padding_mask(src_batch, src_pad_idx)
        memory_mask = src_key_pad_mask.unsqueeze(1).expand(-1, tgt_input.size(1), -1)

        subsequent_mask, tgt_key_pad_mask = make_tgt_masks(tgt_input, tgt_pad_idx)
        causal = subsequent_mask.unsqueeze(0).expand(tgt_input.size(0), -1, -1)
        tgt_mask = causal | tgt_key_pad_mask.unsqueeze(2)

        optimizer.zero_grad()
        logits = model(
            src_batch,
            tgt_input,
            src_mask=None,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )
        loss = criterion(logits.view(-1, logits.size(-1)), tgt_target.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, src_pad_idx, tgt_pad_idx, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for src_batch, tgt_batch in dataloader:
            tgt_input = tgt_batch[:, :-1]
            tgt_target = tgt_batch[:, 1:]
            src_key_pad_mask = make_src_key_padding_mask(src_batch, src_pad_idx)
            memory_mask = src_key_pad_mask.unsqueeze(1).expand(
                -1, tgt_input.size(1), -1
            )

            subsequent_mask, tgt_key_pad_mask = make_tgt_masks(tgt_input, tgt_pad_idx)
            causal = subsequent_mask.unsqueeze(0).expand(tgt_input.size(0), -1, -1)
            tgt_mask = causal | tgt_key_pad_mask.unsqueeze(2)

            logits = model(
                src_batch,
                tgt_input,
                src_mask=None,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
            )
            loss = criterion(logits.view(-1, logits.size(-1)), tgt_target.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)


def greedy_decode(
    model, src_sentence_tensor, src_vocab, tgt_vocab, tgt_itos, max_len=50, device="cpu"
):
    model.eval()
    pad_idx = src_vocab[PAD]
    src_key_pad_mask = make_src_key_padding_mask(src_sentence_tensor, pad_idx)
    memory_mask = src_key_pad_mask.unsqueeze(1)

    with torch.no_grad():
        memory = model.encode(src_sentence_tensor.to(device), src_mask=None)
        ys = torch.tensor([[tgt_vocab[BOS]]], dtype=torch.long, device=device)
        for i in range(max_len - 1):
            subsequent_mask, tgt_key_pad_mask = make_tgt_masks(ys, tgt_vocab[PAD])
            causal = subsequent_mask.unsqueeze(0).expand(ys.size(0), -1, -1)
            tgt_mask = causal | tgt_key_pad_mask.unsqueeze(2)
            out = model.decode(
                ys, memory, tgt_mask=tgt_mask, memory_mask=memory_mask.to(device)
            )
            prob = model.output_proj(out[:, -1, :])
            next_word = torch.argmax(prob, dim=-1).item()
            ys = torch.cat(
                [ys, torch.tensor([[next_word]], dtype=torch.long, device=device)],
                dim=1,
            )
            if next_word == tgt_vocab[EOS]:
                break
    return [tgt_itos[i] if i in tgt_itos else "<unk>" for i in ys.squeeze(0).tolist()]


# ---------- Main script ----------
def transformer_train(device_str="cuda" if torch.cuda.is_available() else "cpu"):
    device = torch.device(device_str)
    print(f"Using device: {device}")

    random.seed(42)
    torch.manual_seed(42)

    print("Loading dataset from CSV files...")
    dataset = load_csv_data(
        train_path="/home/yhhe/project/Deep_Learning/NLP/datasets/tatoeba/train_zh_en.csv",
        test_path="/home/yhhe/project/Deep_Learning/NLP/datasets/tatoeba/test_zh_en.csv",
    )

    # 分割验证集
    split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
    dataset = DatasetDict(
        {
            "train": split_dataset["train"],
            "valid": split_dataset["test"],
            "test": dataset["test"],
        }
    )

    print("Building vocabularies...")

    (src_vocab, src_itos), (tgt_vocab, tgt_itos) = build_vocabs(dataset, min_freq=2)

    print(f"Vocab sizes -> SRC: {len(src_vocab)}, TGT: {len(tgt_vocab)}")

    # Hyperparams
    NUM_LAYERS = 2
    EMBED_DIM = 256
    NUM_HEADS = 4
    FF_DIM = 512
    BATCH_SIZE = 64
    N_EPOCHS = 10
    LR = 1e-3

    src_pad_idx = src_vocab[PAD]
    tgt_pad_idx = tgt_vocab[PAD]

    # 注意：HuggingFace Dataset 支持被 DataLoader 使用（每个 item 是 dict）
    train_loader = DataLoader(
        dataset["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, src_vocab, tgt_vocab, device),
    )
    valid_loader = DataLoader(
        dataset["valid"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, src_vocab, tgt_vocab, device),
    )
    test_loader = DataLoader(
        dataset["test"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, src_vocab, tgt_vocab, device),
    )

    model = TransformerModel(
        len(src_vocab),
        len(tgt_vocab),
        NUM_LAYERS,
        EMBED_DIM,
        NUM_HEADS,
        FF_DIM,
        dropout=0.1,
        pad_idx=src_pad_idx,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_valid_loss = float("inf")
    for epoch in range(1, N_EPOCHS + 1):
        start_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, src_pad_idx, tgt_pad_idx, device
        )
        valid_loss = evaluate(
            model, valid_loader, criterion, src_pad_idx, tgt_pad_idx, device
        )
        end_time = time.time()
        print(
            f"Epoch: {epoch:02} | Train Loss: {train_loss:.4f} | Val Loss: {valid_loss:.4f} | Time: {(end_time - start_time):.2f}s"
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "best_transformer_mt.pt")
            print("\tSaved best model.")

    # Test greedy decode
    model.load_state_dict(torch.load("best_transformer_mt.pt", map_location=device))
    model.to(device)
    test_sample = dataset["test"].select(range(min(3, len(dataset["test"]))))
    for example in test_sample:
        src_sent = example["translation"][SRC_LANG]
        tgt_sent = example["translation"][TGT_LANG]
        src_tok = tokenize_basic(src_sent)
        src_idx = torch.tensor(
            [
                [src_vocab[BOS]]
                + [src_vocab.get(t, src_vocab[UNK]) for t in src_tok]
                + [src_vocab[EOS]]
            ],
            dtype=torch.long,
        ).to(device)
        pred_tokens = greedy_decode(
            model, src_idx, src_vocab, tgt_vocab, src_itos, max_len=50, device=device
        )
        # pred_tokens 已经是 token 列表
        if EOS in pred_tokens:
            pred_tokens = pred_tokens[1 : pred_tokens.index(EOS)]
        else:
            pred_tokens = pred_tokens[1:]
        print("SRC:", src_sent)
        print("REF:", tgt_sent)
        print("PRED:", " ".join(pred_tokens))
        print("-" * 40)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use, e.g. 'cpu', 'cuda', 'cuda:0'",
    )
    args = parser.parse_args()
