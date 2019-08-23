# file handling, general
from datetime import datetime
import simplejson
import pathlib
import string
import io

# nltk 
from nltk.corpus import stopwords

# gensim
import gensim
import gensim.corpora as corpora
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, Phraser
from gensim.summarization.textcleaner import split_sentences

# spacy 
import spacy

### File import ###

def get_files(path):
	path_dir = pathlib.Path(path)
	file_list = []
	for item in path_dir.iterdir():
		if item.is_file():
			file_list.append(item.name)
	return file_list       

def build_one(file, path):
	'''
	Input: Text files from path
	Why: Convert str to list of tokens
	Output: preprocess=True: List of 
		comma-delimited words
			preprocess=False: List of
		comma-delimited sentences
	'''
	with io.open(path.joinpath(file), "rt") as infile:
		return infile.read()

def build_texts(path):
	'''
	Input: Path containing text files
	Why: Prepare corpus docs for gensim
	Output: 2 Lists: List of lists, 
		one for each doc,
		of comma-delimited words;
			List of lists,
		one for each doc, of comma-delimited
		sentences
	'''
	raw_texts = []
	processed_texts = []
	path = pathlib.Path(path)
	for file in get_files(path):
	    text = build_one(file, path)
	    raw_texts.append(split_sentences(text))
	    processed_texts.append(gensim.utils.simple_preprocess(text, deacc=True, min_len=3)) #  preprocess=preprocess
	return raw_texts, processed_texts

### Preprocessing ###

def remove_stops(texts):
		'''
		Input: List of lists, one for 
		each corpus doc, of comma-delimited words
		Why: Remove information-poor words
		Output: List of lists, one for each doc, 
			of non-stopwords in corpus
		'''
		stops = stopwords.words("english")

		return [[word for word in simple_preprocess(str(text)) if word not in stops
		and word not in string.punctuation] for text in texts]

def build_ngrams(texts):
	'''
	Input: List of lists, one for each doc,
		of comma-delimited words
	Why: Build n-grams for words in corpus
	Output: Two lists of lists, bigrams and trigrams;
		bigrams is a proper subset of trigrams
	'''
	bigram_phrase = gensim.models.Phrases(texts, min_count=2, threshold=100)
	trigram_phrase = gensim.models.Phrases(bigram_phrase[texts], threshold=100)

	bigram_model = gensim.models.phrases.Phraser(bigram_phrase)
	trigram_model = gensim.models.phrases.Phraser(trigram_phrase)

	bigrams = [bigram_model[text] for text in texts]
	trigrams = [trigram_model[bigram_model[text]] for text in texts]

	return bigrams, trigrams


def lemmatize(texts, pos_tags = ["NOUN", "ADJ", "VERB", "ADV"], filename="lemmatized_corpus"):
	'''
	Input: Preprocessed list of lists, one for each doc,
		of comma-delimited words
	Why: Lemmatize tokens to remove redundancy
	Output: List of lists of lemmas
	'''
	nlp = spacy.load('en', disable=['parser', 'ner'])
	lemmata = []
	for t_list in texts:
	    doc = nlp(" ".join(t_list)) 
	    lemmata.append([token.lemma_ for token in doc if token.pos_ in pos_tags])
	
	######
	lem_path = pathlib.Path(filename + ".txt")
	i = 1
	while lem_path.exists():
	    lem_path = pathlib.Path(filename + f"{i}.txt")
	    i += 1
	with open(str(lem_path), "w") as f:
		simplejson.dump(lemmata, f)
	directory = pathlib.Path.cwd()
	new_lem_path = max((f.stat().st_mtime, f) for f in directory.iterdir())
	print(f"Lemmatized corpus saved at {new_lem_path[1]}.")
	######

	return lemmata

def make_dictionary(texts, filename="test_docs"):
	'''
	Input: Preprocessed (lemmatized) texts
	Why: Get dictionary and corpus to create
		models in gensim
	Output: Gensim Dictionary object

	'''
	dictionary = Dictionary(lem_texts)

	######
	dict_path = pathlib.Path(filename + ".dict")
	i = 1
	while dict_path.exists():
	    dict_path = pathlib.Path(filename + f"{i}.dict")
	    i += 1
	dictionary.save(str(dict_path))
	directory = pathlib.Path.cwd()
	newdict_path = max((f.stat().st_mtime, f) for f in directory.iterdir())
	print(f"New dictionary saved at {newdict_path[1]}.")
	######

	return dictionary

if __name__ == "__main__":

	texts = build_texts("test docs")
	stopped_texts = remove_stops(texts)
	_, tri_texts = build_ngrams(stopped_texts) #trigrams include all bigrams
	lem_texts = lemmatize(tri_texts, pos_tags = ["NOUN", "ADJ", "VERB", "ADV"])
	dictionary = make_dictionary(lem_texts)