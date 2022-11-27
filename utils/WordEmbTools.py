import os
import re
import numpy as np
import string
import gensim
import nltk
from nltk import word_tokenize, pos_tag
from nltk.tag.mapping import tagset_mapping
from nltk.stem.snowball import EnglishStemmer as Stemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('tagsets')
nltk.download('universal_tagset')

#### Class ################################################

class WordEmbTools(object):
	def __init__(self, CF, LOAD_EMB=False):

		# Embedding models path (checkpoint)
		self.IN_PATH_EMB_CKPT = CF.PATH_EMB_CKPT

		# English NLP
		self._stemmer = Stemmer()
		self.stopwords = set(stopwords.words('english'))

		#### INIT ###################

		self.model = None
		self.zerosVec = []

		# PoS tag map (All)
		self.dpost = {}
		for i, tag in enumerate(sorted(set(tagset_mapping('en-ptb', 'universal').values()))):
			self.dpost[tag] = i

		# Load embedding model?
		if LOAD_EMB:
			self.loadModel()

	#### Functions ############################################

	def loadModel(self):
		if os.path.isfile(self.IN_PATH_EMB_CKPT):

			self.model = gensim.models.FastText.load(self.IN_PATH_EMB_CKPT)

			self.zerosVec = np.zeros(self.model.vector_size)

			print("Fasttext Embedding Shape: {}".format(self.model.wv.vectors.shape))
		else:
			print("Warning: Model not trained yet.")

	# Tokenize (tokens) & PoS
	def word_pos_tokenize(self, txt):
		for token in self.nlp(txt):
			yield (token.text, token.pos_)

	# Preprocess single word
	def preprocWord(self, w):
		w = w.strip().lower()
		w = self.rep_nums(w)
		w = self.rem_punctuation(w)

		if w not in self.stopwords:
			w = self._stemmer.stem(w)
		else:
			w = ""

		return w

	# Preprocess list of words
	def preprocWords(self, words, empty=0):
		rwords = []
		for w in words:
			# - preprocess word
			w = self.preprocWord(w)
			
			# empty: return empty words?
			if empty or w:
				rwords.append(w)

		return rwords

	# Tokenize & preprocess text
	def preprocTxt(self, text, pos=0, empty=0):
		rwords = []
		rtags = []

		# - tokenize
		tokens = word_tokenize(text)
		for w in tokens:

			# - preprocess word
			w = self.preprocWord(w)

			# empty: return empty words?
			if empty or w:
				rwords.append(w)

		if pos:
			rtags = [self.dpost[x[1]] for x in pos_tag(tokens, tagset='universal')]
			return rwords, rtags
		return rwords

	# Average embedding
	def AvgEmbedding(self, prewords):
		if prewords:
			return self.model.wv.get_mean_vector(prewords)
		return self.zerosVec

	# Single word embedding
	def WordEmbedding(self, preword):
		return self.model.wv[preword]

	# Sequence of word embeddings
	def SeqEmbedding(self, prewords):
		seq = []
		for preword in prewords:
			seq.append(self.model.wv[preword])
		return seq

	# Replace (used for preprocessing)
	def _replace(self, pattern, code, text, pos):
		m = re.compile(pattern).search(text, pos)
		if m is not None:
			start = m.span()[0]
			end = m.span()[1]
			return True, text[:start] + code + text[end:], start
		else:
			return False, text, pos

	# Replace (used for preprocessing)
	def replace(self, pattern, code, text):
		test = True
		pos = 0
		while test:
			test, text, pos = self._replace(pattern, code, text, pos)
		return text

	# Replace numbers (used for preprocessing)
	def rep_nums(self, text):
		text = re.sub('(\d+\.\d+)', 'FLOAT', text)
		text = re.sub('(\d+)', 'INT', text)
		return text

	# Replace punctuation (used for preprocessing)
	def rem_punctuation(self, text):
		return text.translate(str.maketrans('', '', string.punctuation))
