import spacy
    
nlp = spacy.load("en_core_web_sm")
doc = nlp("X is buying U.K. firm for $1.")

print([token.text for token in doc])