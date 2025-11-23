import spacy
    
nlp = spacy.load("en_core_web_sm")
doc = nlp("X is buying U.K. firm for $1.")

for token in doc:
    print(token.text, token.pos_, token.dep_)