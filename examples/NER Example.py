import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple has $1 billion sales in U.K. in September 2025.")

for ent in doc.ents:
    print(ent.text, ent.label_)