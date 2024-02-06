import gradio as gr
from transformers import pipeline

import langid

def pretrained_models_translation(input_text):
    lang, confidence = langid.classify(input_text)

    print(f"Lingua rilevata: {lang}")
    print(f"Confidenza: {confidence}")

    result = None
    match lang:
        case "fr":
            print("Loading language translator fr-en...")
            pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
            result = pipe(input_text)
        case "nl":
            print("Loading language translator nl-en...")
            pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-nl-en")
            result = pipe(input_text)
        case "pt":
            print("Loading language translator pt-en...")
            pipe = pipeline("translation", model="unicamp-dl/translation-pt-en-t5")
            result = pipe(input_text)
        case "fi":
            print("Loading language translator fi-en...")
            pipe = pipeline("translation", model="Helsinki-NLP/opus-tatoeba-fi-en")
            result = pipe(input_text)
        case "it":
            print("Loading language translator it-en...")
            pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-it-en")
            result = pipe(input_text)
        case "pl":
            print("Loading language translator pl-en...")
            pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-pl-en")
            result = pipe(input_text)
        case "ru":
            print("Loading language translator ru-en...")
            pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-ru-en")
            result = pipe(input_text)
        case "de":
            print("Loading language translator de-en...")
            pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-de-en")
            result = pipe(input_text)
        case _:
            return "Language not correctly identified or not available!"
    
    print(result)
    print(result[0]['translation_text'])

    return result[0]['translation_text']

if __name__ == '__main__':
    interface = gr.Interface(fn=pretrained_models_translation, inputs=gr.Textbox(lines=2, placeholder='Text to translate'), outputs='text')
    interface.launch()