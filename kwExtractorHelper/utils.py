import spacy
import os
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
import argparse
import shutil
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import string
from nltk.util import ngrams
from rake_nltk import Rake
import yake
nlp = spacy.load("en_core_web_sm")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

def is_noun_phrase(phrase):
    words = nltk.word_tokenize(phrase)
    pos_tags = nltk.pos_tag(words)
    # Check if all words are either nouns (NN, NNS, NNP, NNPS) or adjectives (JJ)
    return all(tag.startswith('NN') or tag == 'JJ' for word, tag in pos_tags)


class setExtractor(object):
	"""docstring for kwExtractor"""
	def __init__(self, arg):
		super(kwExtractor, self).__init__()
		self.arg = arg

	def kw_spacy(doc, output=None, keep=None):
		if keep is None:
			return list([np.text.lower() for np in doc.noun_chunks])
		if output is None:
			output = {}
		kws = []
		for nc in doc.noun_chunks:
			ws = []
			for word in nc:
				if (word.pos_ in keep) and (len(word) > 2):
					ws.append(word.text.lower())
			if len(ws) > 0:
				n = ' '.join(ws)
				output[n] = output.get(n, 0) + 1
				kws.append(n)
		return output, kws

	def kw_NLTK(text, keep=None):
		sentences = nltk.sent_tokenize(text)
		tagged_sentences = [nltk.pos_tag(nltk.word_tokenize(sentence)) for sentence in sentences]

		grammar = "NP: {<JJ>*<NN>*}"

		cp = nltk.RegexpParser(grammar)

		# Extract noun phrases from each sentence
		noun_phrases = []
		for tagged_sentence in tagged_sentences:
			tree = cp.parse(tagged_sentence)
			for subtree in tree.subtrees():
				if subtree.label() == 'NP':
					noun_phrases.append(' '.join(word for word, pos in subtree.leaves()))

		# Count the frequency of each noun phrase
		noun_phrase_freq = Counter(noun_phrases)
		return noun_phrase_freq, noun_phrases

	def kw_RAKE(text, keep=None):
		rake = Rake()
		rake.extract_keywords_from_text(text)

		ranked_phrases = rake.get_ranked_phrases()
		noun_phrases = [phrase for phrase in ranked_phrases if is_noun_phrase(phrase)]

		noun_phrase_freq = Counter(noun_phrases)
		return noun_phrase_freq, noun_phrases

	def kw_YAKE(text, keep=None):
		yake_extractor = yake.KeywordExtractor()

		keywords = yake_extractor.extract_keywords(text)

		noun_phrases = [phrase for phrase, score in keywords if is_noun_phrase(phrase)]

		# Count the frequency of each noun phrase
		noun_phrase_freq = Counter(noun_phrases)
		return noun_phrase_freq, noun_phrases

def mkdir(idir):
	if not os.path.isdir(idir):
		os.makedirs(idir)

def increase_count(idict, key, freq):
	if key not in idict:
		idict[key] = 0
	idict[key] += freq


def add_to_dict(idict, key, value, freq=1):
	if key not in idict:
		idict[key] = {}
	if value not in idict[key]:
		idict[key][value] = 0
	idict[key][value] += freq


def get_unique_values(idict, count_only=False):
	if count_only:
		return {k: len(set(v)) for k, v in idict.items()}
	else:
		return {k: list(set(v)) for k, v in idict.items()}


def save_np_info(np2count, np2reviews, np2rest, np2users, ofile, count_only=False):
	output = {"np2count": np2count, "np2reviews": np2reviews,
			'np2rests': np2rest, 'np2users': np2users}
	json.dump(output, open(ofile, 'w'))
	print("Saved to", ofile)


def extract_raw_keywords_for_reviews(data, ofile, keep=['ADJ', 'NOUN', 'PROPN', 'VERB'],
									 overwrite=True, review2keyword_ofile=None, argsExtractor="kw_spacy"):
	if os.path.isfile(ofile) and not overwrite:
		print("Existing output file. Stop! (set overwrite=True to overwrite)")
		return
	np2count = {}   # frequency
	np2review2count = {}  # reviews
	np2rest2count = {}  #
	np2user2count = {}
	review2keywords = {}
	for rid, uid, restid, text in tqdm(zip(data['review_id'],
		data['user_id'], data['rest_id'], data['text']), total=len(data)):
		doc = text
		if argsExtractor == "kw_spacy":
			doc = nlp(text)
		kwExtractor = getattr(setExtractor, argsExtractor)
		tmp, keywords = kwExtractor(doc, keep=keep)  # np for this review
		for np, freq in tmp.items():
			increase_count(np2count, np, freq)
			add_to_dict(np2review2count, np, rid, freq)
			add_to_dict(np2rest2count, np, restid, freq)
			add_to_dict(np2user2count, np, uid, freq)
		review2keywords[rid] = keywords
	save_np_info(np2count, np2review2count, np2rest2count, np2user2count, ofile)
	if review2keyword_ofile is not None:
		df = pd.DataFrame({"Review_ID": list(review2keywords.keys()), "Keywords": list(review2keywords.values())})
		df.to_csv(review2keyword_ofile)


def load_split(sfile='./data/reviews/splits.json', city='singapore', setname='test'):
	if city == "tripAdvisor":
		sfile = './data/reviews/tripAdvisor_splits.json'
	return json.load(open(sfile))[city][setname]


def filter_keywords(ifile, ofile, min_freq=3):
	data = json.load(open(ifile))
	np2count = data['np2count']
	valid_kws = [a for a, b in np2count.items() if b >= min_freq]
	new_dict = {}
	for k, v in data.items():
		tmp = {}
		for k2 in valid_kws:
			tmp[k2] = v[k2]
		new_dict[k] = tmp
	json.dump(new_dict, open(ofile, 'w'))
	print("Saved to", ofile)


def group_keywords_for_users(ifile, ofile):
	dt = json.load(open(ifile))
	np2users = dt['np2users']
	u2kw = {}  # {user: {keyword: freq}}
	for kw, u2c in np2users.items():
		for u, c in u2c.items():
			if u not in u2kw:
				u2kw[u] = {}
			u2kw[u][kw] = c
	json.dump(u2kw, open(ofile, 'w'))
	print("Saved to", ofile)


def group_keywords_for_rests(ifile, ofile):
	dt = json.load(open(ifile))
	np2rests = dt['np2rests']
	u2kw = {}  # {rest: {keyword: freq}}
	for kw, u2c in np2rests.items():
		for u, c in u2c.items():
			if u not in u2kw:
				u2kw[u] = {}
			u2kw[u][kw] = c
	json.dump(u2kw, open(ofile, 'w'))
	print("Saved to", ofile)


def compute_tfirf(ifile, ofile, irf, default_irf=0.01, sorting=True):
	dt = json.load(open(ifile))
	u2kw2score = {}
	for u, kw2f in dt.items():
		kw2score = {}
		for kw, f in kw2f.items():
			kw2score[kw] = f * irf.get(kw, default_irf)
		u2kw2score[u] = kw2score
	# sort
	if sorting:
		tmp = {}
		for k, v in u2kw2score.items():
			vs = sorted(v.items(), key=lambda x: x[1], reverse=True)
			tmp[k] = vs
		u2kw2score = tmp
	json.dump(u2kw2score, open(ofile, 'w'))
	print("Saved to", ofile)


def get_irf(city, irf_dict, irf_dir):
	if city not in irf_dict:
		irf = json.load(open(os.path.join(irf_dir, city)))
		irf_dict[city] = irf
	return irf_dict[city]


def compute_irf(num, N=1000):
	return np.log(N / num)


def compute_irf_for_dir(idir, odir, N=1000):
	for fname in os.listdir(idir):
		# print(fname)
		if fname.startswith(".") or not fname.endswith(".json"):
			continue
		# print(fname)
		ifile = os.path.join(idir, fname)
		# print(ifile)
		ofile = os.path.join(odir, fname)
		dt = json.load(open(ifile))['np2rests']
		np2irf = {}
		for n, r in dt.items():
			np2irf[n] = compute_irf(len(r), N=N)
		json.dump(np2irf, open(ofile, 'w'))
		print("Saved to", ofile)


def compute_iUf_for_dir(idir, odir, N=1000):
	for fname in os.listdir(idir):
		# print(fname)
		if fname.startswith(".") or not fname.endswith(".json"):
			continue
		# print(fname)
		ifile = os.path.join(idir, fname)
		# print(ifile)
		ofile = os.path.join(odir, fname)
		dt = json.load(open(ifile))['np2users']
		np2iuf = {}
		for n, r in dt.items():
			np2iuf[n] = compute_irf(len(r), N=N)
		json.dump(np2iuf, open(ofile, 'w'))
		print("Saved to", ofile)


def compute_tfiuf(ifile, ofile, irf, default_irf=0.01, sorting=True):
	dt = json.load(open(ifile))
	u2kw2score = {}
	for u, kw2f in dt.items():
		kw2score = {}
		for kw, f in kw2f.items():
			kw2score[kw] = f * irf.get(kw, default_irf)
		u2kw2score[u] = kw2score
	# sort
	if sorting:
		tmp = {}
		for k, v in u2kw2score.items():
			vs = sorted(v.items(), key=lambda x: x[1], reverse=True)
			tmp[k] = vs
		u2kw2score = tmp
	json.dump(u2kw2score, open(ofile, 'w'))
	print("Saved to", ofile)