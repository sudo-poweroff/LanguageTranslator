#Pretrained model
from transformers import pipeline

from torch.utils.tensorboard import SummaryWriter

#metrics
import torchmetrics
from torchmetrics.text import BLEUScore
import nltk
from nltk.translate import meteor_score

from pathlib import Path
import os
import json

def compute_metrics(writer, predicted, expected, src_lang):
    # Evaluate the character error rate
    # Compute the char error rate 
    metric = torchmetrics.CharErrorRate()
    cer = metric(predicted, expected)
    writer.add_scalar('CER pretrained model '+src_lang+'_en', cer)
    writer.flush()

    # Compute the word error rate
    metric = torchmetrics.WordErrorRate()
    wer = metric(predicted, expected)
    writer.add_scalar('WER pretrained model '+src_lang+'_en', wer)
    writer.flush()

    # Compute the BLEU metric
    metric = BLEUScore()
    # Create a list of list to compute BLEU (only for the expected sentences)
    expected_bleu = list()
    for sentece in expected:
        expected_bleu.append([sentece])
    
    bleu_value = metric(predicted, expected_bleu)
    writer.add_scalar('BLEU pretrained model '+src_lang+'_en', bleu_value)
    writer.flush()

    # Compute the METEOR metric
    meteor_scores=list()
    for exepectation, prediction in zip(expected, predicted):
        exepectation= nltk.word_tokenize(exepectation)
        prediction= nltk.word_tokenize(prediction)
        score = meteor_score.single_meteor_score(exepectation, prediction)
        meteor_scores.append(score)
    average_meteor_score= sum(meteor_scores) / len(meteor_scores)

    writer.add_scalar('METEOR pretrained model '+src_lang+'_en', average_meteor_score)
    writer.flush()

    #Print metrics
    print("Evaluation metrics on "+src_lang+"_en:")
    print("CER: {}, WER: {}, BLEU: {}, METEOR: {}".format(cer, wer, bleu_value, average_meteor_score))

def get_test_data(path_test_data:Path):
    data = None
    with open(path_test_data, "r", encoding='utf-8') as json_file:
        data = json.load(json_file)
    
    return data

def test_model(data, src_lang, writer):
    predicted = list()
    expected = list()
    count = 1
    match src_lang:
        case "fr":
            print("Loading language translator fr-en...")
            pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
        case "nl":
            print("Loading language translator nl-en...")
            pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-nl-en")
        case "pt":
            print("Loading language translator pt-en...")
            pipe = pipeline("translation", model="unicamp-dl/translation-pt-en-t5")
        case "fi":
            print("Loading language translator fi-en...")
            pipe = pipeline("translation", model="Helsinki-NLP/opus-tatoeba-fi-en")
        case "it":
            print("Loading language translator it-en...")
            pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-it-en")
        case "pl":
            print("Loading language translator pl-en...")
            pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-pl-en")
        case "ru":
            print("Loading language translator ru-en...")
            pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-ru-en")
        case "de":
            print("Loading language translator de-en...")
            pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-de-en")
    
    missed_translation = 0
    for corpus in data:
        try:
            result = pipe(corpus["translation"][src_lang])
            predicted.append(result[0]['translation_text'])
            expected.append(corpus["translation"]["en"])

            count += 1
            if count % 20 ==0:
                print("Siamo alla frase: ",count)
        except IndexError:
            missed_translation += 1
            print("Missed translation!")
        
    compute_metrics(writer, predicted, expected, src_lang)

    writer.add_scalar('Missed translation '+src_lang+'_en', missed_translation)
    writer.flush()

    return predicted, expected


if __name__ == '__main__':
    writer = SummaryWriter("pretrained_model_metrics")

    print("Testing pretrained language translator on test data...")
    
    #Get test data file name
    list_test_data_filename = os.listdir("test_data")
    for test_file_name in list_test_data_filename:
        print("Loading {} file...".format(test_file_name))
        test_data = get_test_data(Path("test_data/"+test_file_name))

        src_lang_text = test_file_name.split("_")[1]
        predicted, expected = test_model(test_data, src_lang_text, writer)