from pathlib import Path

def get_config(m_folder, lang_src, experiment_name):
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,
        "d_h": 512,
        "datasource": 'opus_books',
        "lang_src": lang_src,
        "lang_tgt": "en",
        "model_folder": m_folder+"_weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_path": 'tokenizers/'+m_folder,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/"+experiment_name
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_weights/{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_weights/{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])