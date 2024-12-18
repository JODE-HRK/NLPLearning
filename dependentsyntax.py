import spacy
sentence = "I have seldom heard him mention her under any other name."

nlp = spacy.load('en_core_web_sm')
doc = nlp(sentence)

for token in doc:
    # print(token.text, "\t", token.dep_, "\t", spacy.explain(token.dep_))
    print(token.text)

    anncestors = [t.text for t in token.ancestors]
    print(anncestors)

for token in doc:
    print(token.text)
    children = [t.text for t in token.children]
    print(children)