



from datasets import load_dataset
import spacy
from spacy.lang.de.examples import sentences

nlp = spacy.load('de_core_news_sm')
doc = nlp(sentences[0])
print(doc.text)
for token in doc:
    print(token.text)

dataset = load_dataset("wmt16" , 'de-en')

pass