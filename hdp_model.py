# gensim
from gensim.corpora import Dictionary
from gensim.models import HdpModel, TfidfModel

# spaCy
import spacy

# general
import re
import pandas as pd
import pathlib
import simplejson
from subprocess import call
from collections import defaultdict 
from datetime import datetime   

# preprocessing
import preprocessing as pp
import tweets

## init from disk
def make_corpus(tfidf=True):
	'''
	Input: Saved dictionary from file
	Why: Load dictionary and create corpus to feed
		into gensim models
	Output: Returns the loaded dictionary object
		and a corpus, either tfidf or doc2bow
	'''
	directory = pathlib.Path.cwd()
	dicts = sorted([str(path) for path in directory.iterdir() if path.is_file() and path.suffix == ".dict"])
	corpora = sorted([str(path) for path in directory.iterdir() if path.is_file() and "lemmatized_corpus" in path.name])
	dict_path = dicts[0]
	corp_path = corpora[0]
	dictionary = Dictionary.load(dict_path)
	with open(corp_path, "r") as rf:
		corpus = simplejson.load(rf)

	bow_corpus = [dictionary.doc2bow(text) for text in corpus]
	if tfidf==True:
		#bow_corpus = [dictionary.doc2bow(text) for text in corpus]
		tfidfmodel = TfidfModel(bow_corpus, smartirs='ntc') 
		tfidf_corpus = tfidfmodel[bow_corpus]
		print("Creating corpus using TF/IDF.")
		return dictionary, tfidf_corpus
	else:
		print("Creating corpus using Doc2BoW.")
		#bow_corpus = [dictionary.doc2bow(text) for text in corpus]
		return dictionary, bow_corpus


class HDP:
	'''
	Class: Hierarchical Dirichlet Process model
		for extracting "hashtags"
	Why: HDP allows for online updating; can learn the
		number of topics (i.e., no. of topics is a random
		variable subject to Dirichlet process); and groups 
		may share mixture components. TFIDF and POS tags 
		are employed as additional relevance/salience metrics.
		the random distribution of atoms paired with a small corpus 
		ensure a measure of serendipity which is advantageous
		in the "hashtag" context, where latent/non-obvious/
		fortuitous thematic relationships are desirable to
		uncover.
	To improve: Add pre-trained embeddings (e.g. GloVe, wiki corpus)
		for transfer learning to augment performance of model on small
		corpora; incorporate coherence measures to adjudge topic
		salience
	'''
	def __init__(self, dictionary, corpus, num_topics=10, num_words=7, pos_tags=["NOUN", "ADJ"]):
		
		self.num_topics = num_topics
		self.num_words = num_words
		self.pos_tags = pos_tags
		self.dictionary, self.corpus = dictionary, corpus
		self.model = HdpModel(corpus=self.corpus, id2word=self.dictionary)
		
		self.doc_terms_df = self.get_doc_terms(self.model, self.corpus)
		self.doc_terms = [word for row in self.doc_terms_df["Topic Keywords"] for word in row.split(", ")]
		self.important_terms = self.get_important_terms(self.model, num_topics=self.num_topics, num_words=self.num_words, pos_tags=self.pos_tags)
		
		self.raw_texts, _ = pp.build_texts("test docs")
		self.key_sents = self.get_key_sents(self.raw_texts, self.important_terms)
		    
	def get_doc_terms(self, hdpmodel, corpus):
		''' 
		Input: Model, preprocessed corpus
		Why: Get top terms _per document_
		Output: DataFrame containing document-specific
			topics, % contribution, and keywords, for
			use if desired in hashtag generation
		'''
		top_df = pd.DataFrame()
		
		# Get main topic in each document
		for i, row in enumerate(hdpmodel[corpus]):
		    row = sorted(row, key=lambda x: (x[1]), reverse=True)
		    # Get the dominant topic, % contribution and topic keywords for each document
		    for j, (topic, topic_pct) in enumerate(row):
		        if j == 0:  # the first is the dominant topic
		            wp = hdpmodel.show_topic(topic, topn=10)
		            topic_keywords = ", ".join([word for word, pct in wp])
		            top_df = top_df.append(pd.Series([int(topic), round(topic_pct,4), topic_keywords]), ignore_index=True)
		        else:
		            break
		top_df.columns = ['Dominant Topic', 'Percent Contribution', 'Topic Keywords']
		return top_df
		
	def get_important_terms(self, hdpmodel, num_topics, num_words, pos_tags):
		''' 
		Input: Model, number of desired topics, number of desired
			terms per topic, desired part-of-speech tags to allow
			in the final set
		Why: Generate a list of salient terms according to document topics,
			relevant parts of speech, and required number
		Output: List of comma-delimited words, narrowed by POS tag, from
			which to generate hashtags
		'''
		nlp = spacy.load('en', disable=['parser', 'ner'])
		sort = sorted(hdpmodel.show_topics(num_topics=num_topics, num_words=num_words,formatted=False),
		                    key=lambda x: x[0])
		topic_terms = " ".join([term[0] for item in sort for term in item[1]])
		doc = nlp(topic_terms)
		important_words = [token.text for token in doc if token.pos_ in pos_tags]
		return important_words
		
	def get_key_sents(self, raw_texts, key_terms):
		'''
		Input: Topic-model generated salient terms, raw documents
		Why: Get the original texts in which terms occurred
			from the pre-preprocessed corpora
		Output: DataFrame containing keywords, document number,
			and sentences in which terms occur for each provided
			keyword 
		''' 
		raw_sents = [[s for s in text] for text in self.raw_texts]
		keywords = key_terms
		hits = defaultdict(dict)
		for keyword in keywords:
			if "_" in keyword:
				n = keyword.split("_")
				pattern = re.compile(fr"\b{n[0]}.+{n[1]}\b", re.I)
			else:
				pattern = re.compile(fr"\b{keyword}(ing|ed|s|er|able|d|en)?\b", re.I)
			for doc in raw_sents:
				hits.update({keyword+"_"+str(i): (raw_sents.index(doc), s)for i,s in enumerate(doc) if re.search(pattern, s)})
		hits_df = pd.DataFrame(hits).T
		return hits_df

	def display_hashtags(self):
		'''
		Input: Takes in "important"/salient terms and raw texts
		Why: Generate hashtags with doc references and 
			sentences in which terms occur, using serendipity
			as an additional metric of interest
		Output: (1) HTML file; (2) PDF of table; (3) DataFrame
			containing terms, doc refs, and sentences
		'''
		term_df = dict()
		terms = self.important_terms
		
		# Set keys equal to terms
		for term in terms:
		     term_df[term] = defaultdict(list)
		        
		# Set values equal to corresponding documents and 
		# sentences that occur in them
		for term in terms:
			if "_" in term:
			    n = term.split("_")
			    pattern = re.compile(fr"\b{n[0]}(.+)?{n[1]}s?\b", re.I)
			else:
			    pattern = re.compile(fr"\b{term}(ing|ed|s|er|able|d|en)?\b", re.I)
			for text in self.raw_texts:
			    term_df[term][self.raw_texts.index(text)] = [i for i in text if re.search(pattern, i)]
		
		# Create a dataframe . . .              
		new_df = pd.DataFrame()
		for term in term_df.keys():
			doc_s = []
			sent_s = []
			for k,v in term_df[term].items():
				if term_df[term].get(k) == []:
					pass
				else:
					doc_s.append(k)
					sent_s.append(v)
			temp = pd.DataFrame({"term":[term], "documents":[doc_s], "sentences":[sent_s]})
			#print(temp)
			new_df = new_df.append(temp, ignore_index=True)
		
		# . . . and make it pretty
		new_df["documents"] = new_df["documents"].apply(lambda x: str(x)).apply(lambda x: x[1:-1])
		new_df["sentences"] = new_df["sentences"].apply(lambda x: 
		                                                "<br> ".join([s for l in x for s in l]))
		new_df.style.set_properties(**{
		    "white-space": "pre-wrap",
		    "charset":"utf-8"
		})
		filename = "hashtags"
		filepath = pathlib.Path(filename + ".html")
		i = 1
		while filepath.exists():
		    filepath = pathlib.Path(filename + f"_{i}.html")
		    i += 1
		new_df.to_html(filepath, escape=False)
		call(f"wkhtmltopdf {filepath} {str(filepath)[:-5]}.pdf", shell=True)
		return new_df

if __name__ == "__main__":
	dictionary, tfidf_corpus = make_corpus()
	hdp = HDP(dictionary, tfidf_corpus)
	print("Fetching hashtags from corpus . . .")
	df = hdp.display_hashtags()

	directory = pathlib.Path.cwd()
	new_path = max((f.stat().st_mtime, f) for f in directory.iterdir())
	print(f"PDF created at {new_path[1]}.")

	tweets.get_tweets(hdp.important_terms)
