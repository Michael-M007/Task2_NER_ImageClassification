import torch
from transformers import BertTokenizer, BertForTokenClassification

ANIMAL_NAMES = {"dog", "cat", "horse", "cow", "sheep", "elephant", "lion", "tiger", "bear", "zebra"}

MODEL_PATH = "models/ner_model.pth"
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = BertForTokenClassification.from_pretrained("bert-base-cased")
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

def extract_animals_from_text(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs).logits

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    extracted_animals = []
    for token, logit in zip(tokens, outputs[0]):
        if torch.argmax(logit).item() > 0:
            word = token.lstrip("##")
            if word.lower() in ANIMAL_NAMES:
                extracted_animals.append(word.lower())
    return list(set(extracted_animals))

