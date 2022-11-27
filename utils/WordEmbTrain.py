import os
import gensim

class MySentences(object):
	def __init__(self, CF, listfiles):
		self.itera = 0
		self.IN_DIR_PREPROC_CHUNKS = CF.DIR_PREPROC_CHUNKS
		self.listfiles = listfiles
		
	def __iter__(self):
		self.itera += 1
		# - iterate files
		cpf = 0
		for fname in self.listfiles:
			cpf += 1
			print("iter {}: {}/{}       ".format(self.itera, cpf, len(self.listfiles)), flush=True, end="\r")

			fname = fname.rsplit(".", 1)[0]+".txt"

			fr = open(self.IN_DIR_PREPROC_CHUNKS+fname, "r", encoding="UTF8")

			# - iterate preprocessed chunks
			for l in fr:
				if not l.rstrip():
					continue

				sp = l.split("\t", 1)

				prewords = sp[1].rstrip().split(" ")

				# - return preprocessed words (if any)
				if len(prewords) > 0:
					yield prewords

class WordEmbTrain(object):
	def __init__(self, CF, WET):

		# - update Existing Model
		self.UPDATE_EMB = False

		# - parameters object
		self.CF = CF

		# - embedding checkpoint path
		self.OUT_PATH_CKPT = CF.PATH_EMB_CKPT

		# - prepare checkpoint path
		ckptspdir = self.OUT_PATH_CKPT.replace("\\","/").split("/")
		ckptspdir.pop()
		chckptdir = ""
		for sd in ckptspdir:
			chckptdir += sd + "/"
			if not os.path.isdir(chckptdir):
				os.mkdir(chckptdir)

		# - Embedding model
		self.MyEmbModel = gensim.models.FastText

	def train(self, IN_DIR_PDF, list_files_pdf=[]):

		listfiles = list_files_pdf or os.listdir(IN_DIR_PDF)

		# - prepare sentences iterator
		sents = MySentences(self.CF, listfiles)

		print("Training...")

		# - load to update / retrain from scratch?
		model = self.MyEmbModel(vector_size=100, min_count=5, alpha=0.07, epochs=30, workers=4) # epochs=20

		# - get vocab
		model.build_vocab(sents)

		# - start training
		model.train(sents, epochs=model.epochs, total_examples=model.corpus_count)
		# model.init_sims(replace=True)

		# - save checkpoint
		model.save(self.OUT_PATH_CKPT)

		print("Fasttext Embedding Shape: {}".format(model.wv.vectors.shape))


'''

Default MyEmbModel(...):

sentences=None, corpus_file=None, size=100, alpha=0.025, window=5,
min_count=5, max_vocab_size=None, sample=0.001, seed=1, workers=3,
min_alpha=0.0001, sg=0, hs=0, negative=5, ns_exponent=0.75,
cbow_mean=1, hashfxn=<built-in function hash>, iter=5, null_word=0,
trim_rule=None, sorted_vocab=1, batch_words=10000, compute_loss=False,
callbacks=(), max_final_vocab=None

(*) Link: https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec

'''
