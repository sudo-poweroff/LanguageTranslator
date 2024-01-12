from model import build_transformer
from dataset import causal_mask, get_ds, save_test_data, save_train_data, save_valid_data
from config import get_config, get_weights_file_path, latest_weights_file_path

import torch
import torch.nn as nn


import warnings
from tqdm import tqdm
import os
from pathlib import Path

#metrics
import torchmetrics
import nltk
from nltk.translate import meteor_score

from torch.utils.tensorboard import SummaryWriter

def greedy_decode(model, corpus, corpus_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(corpus, corpus_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(corpus).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(corpus_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, corpus_mask, decoder_input, decoder_mask)

        # get next token therefore the last decoder temporal step
        prob = model.project(out[:, -1]) 
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(corpus).fill_(next_word.item()).to(device)], dim=1)

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        i=0
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            if i % 20==0:
                # Print the source, target and model output
                print_msg('-'*console_width)
                print_msg(f"{f'SOURCE: ':>12}{source_text}")
                print_msg(f"{f'TARGET: ':>12}{target_text}")
                print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")
                print_msg('-'*console_width)
            i+=1
              
    
    if writer:
        cer,wer,bleu,avg_meteor=compute_metrics(writer, predicted, expected, global_step)
        print_msg('-'*console_width)
        print_msg(f"{f'Char Error Rate: ':>12}{cer}")
        print_msg(f"{f'Word Error Rate: ':>12}{wer}")
        print_msg(f"{f'BLEU: ':>12}{bleu}")
        print_msg(f"{f'METEOR: ':>12}{avg_meteor}")
        print_msg('-'*console_width)


def run_test(model, test_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, writer):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        i=1
        batch_iterator = tqdm(test_ds, desc=f"Processing Sentence {i:02d}")
        for batch in batch_iterator:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for test"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            if i % 20==0:
                str_1=' Prediction '+i+' '
                new_console_width=int((console_width-len(str_1))/2)
                str_2='-'*new_console_width
                print_msg(str_2+str_1+str_2)
                print_msg(f"{f'SOURCE: ':>12}{source_text}")
                print_msg(f"{f'TARGET: ':>12}{target_text}")
                print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")
                print_msg('-'*console_width)
            i+=1
    
    if writer:
        cer,wer,bleu,avg_meteor=compute_metrics(writer, predicted, expected, test=True)
        str_1=' Test evaluation metrics '
        new_console_width=int((console_width-len(str_1))/2)
        str_2='-'*new_console_width
        print_msg(str_2+str_1+str_2)
        print_msg(f"{f'Char Error Rate: ':>12}{cer}")
        print_msg(f"{f'Word Error Rate: ':>12}{wer}")
        print_msg(f"{f'BLEU: ':>12}{bleu}")
        print_msg(f"{f'METEOR: ':>12}{avg_meteor}")
        print_msg('-'*console_width)


def compute_metrics(writer, predicted, expected, global_step, test:bool=False):
    # Evaluate the character error rate
    # Compute the char error rate 
    metric = torchmetrics.CharErrorRate()
    cer = metric(predicted, expected)
    if test:
        writer.add_scalar('test cer', cer)
    else:
        writer.add_scalar('validation cer', cer, global_step)
    writer.flush()

    # Compute the word error rate
    metric = torchmetrics.WordErrorRate()
    wer = metric(predicted, expected)
    if test:
        writer.add_scalar('test wer', wer)
    else:
        writer.add_scalar('validation wer', wer, global_step)
    writer.flush()

    # Compute the BLEU metric
    metric = torchmetrics.BLEUScore()
    bleu = metric(predicted, expected)
    if test:
        writer.add_scalar('test ', bleu)
    else:
        writer.add_scalar('validation BLEU', bleu, global_step)
    writer.flush()

    # Compute the METEOR metric
    meteor_scores=list()
    for exepectation, prediction in zip(expected, predicted):
        exepectation= nltk.word_tokenize(exepectation)
        prediction= nltk.word_tokenize(prediction)
        score = meteor_score.single_meteor_score(exepectation, prediction)
        meteor_scores.append(score)
    average_meteor_score= sum(meteor_scores) / len(meteor_scores)

    if test:
        writer.add_scalar('average test METEOR', average_meteor_score)
    else:
        writer.add_scalar('average validation METEOR', average_meteor_score, global_step)
    writer.flush()

    return cer, wer, bleu, average_meteor_score

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_h=config['d_h'])
    return model

def train_and_test_model(config):
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

    # Make sure the weights folder exists
    Path(f"{config['datasource']}_weights").mkdir(parents=True, exist_ok=True)
    Path(f"{config['datasource']}_weights/{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_data_path = Path(f"train_data/train_{config['lang_src']}_{config['lang_tgt']}.json").exists()
    valid_data_path = Path(f"valid_data/valid_{config['lang_src']}_{config['lang_tgt']}.json").exists()
    test_data_path = Path(f"test_data/test_{config['lang_src']}_{config['lang_tgt']}.json").exists()

    train_dataloader = None
    test_dataloader = None
    val_dataloader = None
    tokenizer_src = None
    tokenizer_tgt = None

    if not(train_data_path and valid_data_path and test_data_path):
        train_dataloader, test_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
        #Save data
        save_train_data(train_dataloader, config)
        save_valid_data(val_dataloader,config)
        save_test_data(test_dataloader, config)
    else:
        #Load the saved data
        train_dataloader, test_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config, download_data=False)

    # Obtain transformer
    print("Building transformer...")
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    print("------------------------------------------------ START TRAIN ------------------------------------------------")
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            #(B, seq_len, tgt_vocab_size) --> (B * seq_len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of every epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

    print("------------------------------------------------ START TEST ------------------------------------------------")
    run_test(model, test_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), writer)
    


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    source_lang=str(input("Insert the source language (de, es, fr, it, nl, pl, pt, ru):\n"))
    model_folder=str(input("Insert model folder name (model_srclang_tgtlang):\n"))
    experiment_name=str(input("Insert the experiment name:\n"))
    source_lang=source_lang.strip()
    model_folder=model_folder.strip()
    experiment_name=experiment_name.strip()
    config = get_config(model_folder, source_lang, experiment_name)
    train_and_test_model(config)