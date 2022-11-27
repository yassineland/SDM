import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from numpy import argmax
from re import sub
from math import ceil
from time import time
import sys
sys.path.append("../main")
from conf import *
sys.path.append(DIR_SECTDET)
from runCNN1V import runCNN1V
from sklearn.metrics import classification_report

class LearnCorrectTitles(object):

	def __init__(self, CF, WET):

		# Number of words to take
		self.XWORDS = 10
		# Number of classes
		self.N_CLASSES = 2

		# embedding & NLP tools object
		self.WET = WET

		# Input/Output Folders
		self.IN_DIR_AUTO_ANNOT = CF.DIR_AUTO_CHUNKS
		self.IN_DIR_SEM_PROPS = CF.DIR_SEM_CHUNK_PROPS
		self.IN_DIR_PREPROC_CHUNKS = CF.DIR_PREPROC_CHUNKS
		self.IN_DIR_FORM_PROPS = CF.DIR_FORM_CHUNK_PROPS
		self.IN_DIR_MANU_ANNOT = CF.DIR_MANU_ANNOT

		# Folders/Files save
		self.DIR_CKPT_SECTDET = CF.DIR_CKPT_SECTDET
		self.FILE_CKPT_SECTDET = CF.FILE_CKPT_SECTDET
		self.OUT_DIR_PREDS_SECTDET = CF.DIR_PREDS_SECTDET

		#########################################################

		#### PREPARE FILES ####

		self.ckptpath = self.DIR_CKPT_SECTDET + self.FILE_CKPT_SECTDET

		if not os.path.isdir(self.IN_DIR_FORM_PROPS):
			os.mkdir(self.IN_DIR_FORM_PROPS)
		if not os.path.isdir(self.IN_DIR_SEM_PROPS):
			os.mkdir(self.IN_DIR_SEM_PROPS)
		if not os.path.isdir(self.IN_DIR_PREPROC_CHUNKS):
			os.mkdir(self.IN_DIR_PREPROC_CHUNKS)
		if not os.path.isdir(self.IN_DIR_AUTO_ANNOT):
			os.mkdir(self.IN_DIR_AUTO_ANNOT)

		subdirs = ""
		for subdir in CF.DIR_CKPT_SECTDET.replace("\\", "/").rstrip("/").split("/"):
			subdirs += subdir+"/"
			if not os.path.isdir(subdirs):
				os.mkdir(subdirs)

		if not os.path.isdir(self.OUT_DIR_PREDS_SECTDET):
			os.mkdir(self.OUT_DIR_PREDS_SECTDET)

		#### POS TAGS ####

		self.spos = len(self.WET.dpost)

		#### INITIAL GLOBAL VARIABLES ####

		self.list_X = []
		self.list_Y = []
		self.dict_valid = {}

	#### SOME FUNCTIONS ####

	# transform split to features vector(s)
	def getFeatures(self, visfeat, semfeat, posfeat):
		# Is one vector input:
		# - Syntaxic
		x = list(map(float, visfeat.split(",")))
		# - Semantic
		xs = list(map(float, semfeat.split(",")))
		# - PoS
		xp = list(map(lambda vx:int(vx)/(self.spos-1), posfeat.split(",")[:self.XWORDS]))

		# Padding
		npad = self.XWORDS - len(xp)
		if npad:
			xp = xp + [0]*npad
			
		# Fusion
		x.extend(xs)
		x.extend(xp)
		return x

	# load all data
	def load(self, IN_DIR_PDF, list_files_pdf=[]):
		self.list_X = []
		self.list_Y = []
		self.dict_valid = {}
		nb_doc_tit = 0
		nb_doc_notit = 0

		for i, fname in enumerate(list_files_pdf or os.listdir(IN_DIR_PDF)):

			fname = fname.rsplit(".", 1)[0]+".txt"

			# manual annotation required but not found? skip
			if ((IS_TRAIN and TRAIN_MANU) or (not IS_TRAIN and not EVAL_AUTO)) and not os.path.isfile(self.IN_DIR_MANU_ANNOT+fname):
				continue

			# prepare data list for this document
			self.dict_valid[fname] = {"labels":[], "X":[], "Y":[], "pred":[]}

			nb_notit = 0
			nb_tit = 0

			fr = open(self.IN_DIR_AUTO_ANNOT+fname, "r", encoding="utf-8")
			frp = open(self.IN_DIR_FORM_PROPS+fname, "r", encoding="utf-8")
			frsp = open(self.IN_DIR_SEM_PROPS+fname, "r", encoding="utf-8")
			frpp = open(self.IN_DIR_PREPROC_CHUNKS+fname, "r", encoding="utf-8")
			if ((IS_TRAIN and TRAIN_MANU) or (not IS_TRAIN and not EVAL_AUTO)):
				frma = open(self.IN_DIR_MANU_ANNOT+fname, "r", encoding="utf-8")

			while True:
				l = fr.readline()
				lp = frp.readline()
				lsp = frsp.readline()
				lpp = frpp.readline()
				if ((IS_TRAIN and TRAIN_MANU) or (not IS_TRAIN and not EVAL_AUTO)):
					lma = frma.readline()

				if not l:
					break

				# chunk visual features
				visfeat = lp.rstrip()

				# semantic features
				semsp = lpp.split("\t", 1)
				# - PoS
				posfeat = semsp[0]
				# - word embedding
				semfeat = lsp

				# class
				sp = l.rstrip().split(":",1)
				label = sp[1]
				if (IS_TRAIN and TRAIN_MANU) or (not IS_TRAIN and not EVAL_AUTO):
					y = int(lma.strip())
				else:
					y = int(sp[0])

				# generate features & add to data list
				x = self.getFeatures(visfeat, semfeat, posfeat)
				self.list_X.append(x)
				self.list_Y.append(y)
				self.dict_valid[fname]["X"].append(x)
				self.dict_valid[fname]["Y"].append(y)
				self.dict_valid[fname]["labels"].append(label)

				nb_tit += y
				nb_notit += 1-y
			
			nb_doc_tit += nb_tit
			nb_doc_notit += nb_notit

			print(f"Load doc: {i+1}", flush=True, end="\r")

		nb_doc_tot = (nb_doc_tit + nb_doc_notit) or 1
		print("Infos: [Titles: {} ({:.2f}%), Non titles: {} ({:.2f}%), Total: {}]\n".format(nb_doc_tit, (nb_doc_tit/nb_doc_tot)*100, nb_doc_notit, (nb_doc_notit/nb_doc_tot)*100, nb_doc_tot))

		# (*) free memory
		fr.close()
		frp.close()
		frsp.close()
		frpp.close()
		if ((IS_TRAIN and TRAIN_MANU) or (not IS_TRAIN and not EVAL_AUTO)):
			frma.close()

	#### TRAINING ####
	def train(self):
		print("# TRAINING SECTION DETECTOR:\n")

		t0 = time()

		# My model
		self.mymodel = runCNN1V(self.list_X, self.list_Y, self.ckptpath, 1, self.N_CLASSES)

		# (*) free memory
		self.list_X = None
		self.list_Y = None

	#### DETECTION ####
	def detect(self):

		t1 = time()

		self.mymodel = None
		y_pred = []
		y_test = []
		cpf = 0
		for infile, v in self.dict_valid.items():
			cpf += 1
			print("{}/{}".format(cpf, len(self.dict_valid)))

			# Load model
			if not self.mymodel:
				self.mymodel = runCNN1V(v["X"], v["Y"], self.ckptpath, 0, self.N_CLASSES)

			# Prediction
			v["pred"], nprec, nb_tit = self.mymodel.Predict(v["X"], v["Y"])

			# WRITE VALID
			fw = open(self.OUT_DIR_PREDS_SECTDET+infile, "w", encoding="utf-8")
			
			# [pred class (0 or 1)]:[real class (0 or 1)]	 :  [label]
			for i, _ in enumerate(v["Y"]):

				pred = argmax(v["pred"][i])

				y_pred.append(pred)
				y_test.append(v["Y"][i])
				
				strline = "{}:{}\n".format(pred, v["labels"][i])

				fw.write(strline)

		# Show results
		print("Classification Report (compare to {} annotations):".format(["manual", "auto"][EVAL_AUTO]))
		print(classification_report(y_test, y_pred, digits=4, target_names=["non title", "title"]))
