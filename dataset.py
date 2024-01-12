import torch
import json
from torch.utils.data import  Dataset, DataLoader, random_split
from pathlib import Path

# Huggingface datasets and tokenizers
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

# Huggingface datasets
from datasets import load_dataset

class BilingualDataset(Dataset):

    def __init__(self, corpora, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.corpora = corpora
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.corpora)

    def __getitem__(self, idx):
        corpus = self.corpora[idx]
        src_text = corpus['translation'][self.src_lang]
        tgt_text = corpus['translation'][self.tgt_lang]


        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # We will add <s> and </s>
        # We will only add <s>, and </s> only on the label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Add <s> and </s> token
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only <s> token
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only </s> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
    
def causal_mask(size):
    # torch.triu() returns the upper triangular part of the matrix that represents the words that come after
    # The matrix consist of all 0 and only the upper triangular has 1 values. Therefore we capture the words that come after
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    # The matrix consist of all 1 and only the upper triangular has 0 values
    return mask == 0

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):

    Path(f"{config['tokenizer_path']}").mkdir(parents=True, exist_ok=True)

    tokenizer_file=config['tokenizer_file'].format(lang)
    tokenizer_path = Path(f"{config['tokenizer_path']}/{tokenizer_file}")
    tokenizer= None
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        # Split words on whitespace
        tokenizer.pre_tokenizer = Whitespace()
        # Define trainer and special tokens
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config, download_data:bool=True):
    # It only has the train split, so we divide it overselves
    corpora=None
    train_ds_raw =None
    test_ds_raw = None
    val_ds_raw = None

    print("Loading corpora from Hugging face ({} - {})...".format(config['lang_src'], config['lang_tgt']))
    if config['lang_src'] == 'de':
        corpora = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    else:
        print(f"{config['lang_tgt']}-{config['lang_src']}")
        corpora = load_dataset(f"{config['datasource']}", f"{config['lang_tgt']}-{config['lang_src']}", split='train')
        
    if download_data:
        # Split the corpora in train, test and valid
        print("Corpora lenght: ",len(corpora))
        train_ds_size = int(0.7 * len(corpora))
        test_ds_size = int(0.2 * len(corpora))
        val_ds_size = len(corpora) - train_ds_size - test_ds_size
        print('Train size: {}, Validation size {}, Test size: {}'.format(train_ds_size, val_ds_size, test_ds_size))
        train_ds_raw, test_ds_raw, val_ds_raw = random_split(corpora, [train_ds_size, test_ds_size, val_ds_size])
    else:
        print("Loading pre-saved data...")
        with open("train_data/train_"+config['lang_src']+"_"+config['lang_tgt']+".json", "r", encoding='utf-8') as json_file:
            train_ds_raw = json.load(json_file)
        
        with open("valid_data/valid_"+config['lang_src']+"_"+config['lang_tgt']+".json", "r", encoding='utf-8') as json_file:
            val_ds_raw = json.load(json_file)

        with open("test_data/test_"+config['lang_src']+"_"+config['lang_tgt']+".json", "r", encoding='utf-8') as json_file:
            test_ds_raw = json.load(json_file)

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, corpora, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, corpora, config['lang_tgt'])

    

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    test_ds = BilingualDataset(test_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in corpora:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, test_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def save_test_data(test_ds, config):
    print("Saving test data for translation {} - {}...".format(config['lang_src'], config['lang_tgt']))
    Path("test_data").mkdir(parents=True, exist_ok=True)
    file_path="test_data/test_"+config['lang_src']+"_"+config['lang_tgt']+".json"
    test_data=list()
    for batch in test_ds:
        x={"translation": {config['lang_src'] : batch['src_text'][0], config['lang_tgt'] : batch['tgt_text'][0]}}
        test_data.append(x)
    with open(file_path, "w", encoding='utf-8') as json_file:
        json.dump(test_data, json_file, ensure_ascii=False, indent=None)

def save_train_data(train_ds, config):
    print("Saving train data for translation {} - {}...".format(config['lang_src'], config['lang_tgt']))
    Path("train_data").mkdir(parents=True, exist_ok=True)
    file_path="train_data/train_"+config['lang_src']+"_"+config['lang_tgt']+".json"
    train_data=list()
    for batch in train_ds:
        for src_corpus,tgt_corpus in zip(batch['src_text'],batch['tgt_text']):
            x={"translation": {config['lang_src'] : src_corpus, config['lang_tgt'] : tgt_corpus}}
            train_data.append(x)
    with open(file_path, "w", encoding='utf-8') as json_file:
        json.dump(train_data, json_file, ensure_ascii=False, indent=None)


def save_valid_data(valid_ds, config):
    print("Saving validation data for translation {} - {}...".format(config['lang_src'], config['lang_tgt']))
    Path("valid_data").mkdir(parents=True, exist_ok=True)
    file_path="valid_data/valid_"+config['lang_src']+"_"+config['lang_tgt']+".json"
    valid_data=list()
    for batch in valid_ds:
        x={"translation": {config['lang_src'] : batch['src_text'][0], config['lang_tgt'] : batch['tgt_text'][0]}}
        valid_data.append(x)
    with open(file_path, "w", encoding='utf-8') as json_file:
        json.dump(valid_data, json_file, ensure_ascii=False, indent=None)