import langid
import torch
import gradio as gr

from train import get_model, greedy_decode
from dataset import get_ds
from config import get_config, latest_weights_file_path


device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
print("Using device:", device)
device = torch.device(device)

def classify_language_and_get_tokenizers(input_text):
    #Classify the source language
    lang, confidence = langid.classify(input_text)

    print(f"Lingua rilevata: {lang}")
    print(f"Confidenza: {confidence}")

    config = None
    tokenizer_src = tokenizer_tgt = None
    match lang:
        case "fr":
            print("Loading language translator fr-en...")
            config = get_config("model_fr_en","fr","")
            config["seq_len"] = 485

            #Loading tokenizer
            _, _, _, tokenizer_src, tokenizer_tgt = get_ds(config, download_data=False)
        case "nl":
            print("Loading language translator nl-en...")
            config = get_config("model_nl_en","nl","")
            config["seq_len"] = 660

            #Loading tokenizer
            _, _, _, tokenizer_src, tokenizer_tgt = get_ds(config, download_data=False)
        case "pt":
            print("Loading language translator pt-en...")
            config = get_config("model_pt_en","pt","")
            config["seq_len"] = 210

            #Loading tokenizer
            _, _, _, tokenizer_src, tokenizer_tgt = get_ds(config, download_data=False)
        case "fi":
            print("Loading language translator fi-en...")
            config = get_config("model_fi_en","fi","")
            config["seq_len"] = 200

            #Loading tokenizer
            _, _, _, tokenizer_src, tokenizer_tgt = get_ds(config, download_data=False)
        case "it":
            print("Loading language translator it-en...")
            config = get_config("model_it_en","it","")
            config["seq_len"] = 350

            #Loading tokenizer
            _, _, _, tokenizer_src, tokenizer_tgt = get_ds(config, download_data=False)
        case "pl":
            print("Loading language translator pl-en...")
            config = get_config("model_pl_en","pl","")
            config["seq_len"] = 560

            #Loading tokenizer
            _, _, _, tokenizer_src, tokenizer_tgt = get_ds(config, download_data=False)
        case "ru":
            print("Loading language translator ru-en...")
            config = get_config("model_ru_en","ru","")
            config["seq_len"] = 350

            #Loading tokenizer
            _, _, _, tokenizer_src, tokenizer_tgt = get_ds(config, download_data=False)
        case "de":
            print("Loading language translator de-en...")
            config = get_config("model_de_en","de","")
            config["seq_len"] = 485

            #Loading tokenizer
            _, _, _, tokenizer_src, tokenizer_tgt = get_ds(config, download_data=False)
        case _:
            return "Language not correctly identified or not available!"
    
    return tokenizer_src, tokenizer_tgt, config

def translation_function(input_text):
    #Get tokenizers
    tokenizer_src, tokenizer_tgt, config = classify_language_and_get_tokenizers(input_text)

    #Define SOS - EOS - PAD token
    sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
    eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
    pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)
    
    #Transform input as a tensor and tokenize it
    input_tokens = tokenizer_src.encode(input_text).ids

    padding_size = config["seq_len"] - len(input_tokens) -2
    if padding_size < 0:
        raise ValueError("Sentence is too long")

    encoder_input = torch.cat(
        [
            sos_token,
            torch.tensor(input_tokens, dtype=torch.int64),
            eos_token,
            torch.tensor([pad_token] * padding_size, dtype=torch.int64),
        ],
        dim=0,
    )

    encoder_mask = (encoder_input != pad_token).unsqueeze(0).unsqueeze(0).int()

    #Build transformer
    print("Loading language translator...")
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    #Loading model weights
    weights_file_name = latest_weights_file_path(config)
    state = torch.load(weights_file_name)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    #Move input on GPU
    encoder_input = encoder_input.unsqueeze(0)
    encoder_input = encoder_input.to(device)
    encoder_mask = encoder_mask.to(device)

    #Provide input to the language translator
    model_translation = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, config["seq_len"], device)

    #Decode model output
    model_translation_text = tokenizer_tgt.decode(model_translation.detach().cpu().numpy())
    print("Model translation is:")
    print(model_translation_text)

    return model_translation_text

if __name__ == '__main__':
    interface = gr.Interface(fn=translation_function, inputs=gr.Textbox(lines=2, placeholder='Text to translate'), outputs='text')
    interface.launch()
    
    

    
