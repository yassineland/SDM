import re
import os
from time import time

#####################################

class CalcSemaProps(object):

	def __init__(self, CF, WET):

		# embedding & NLP tools object
		self.WET = WET

		# input directory of preprocessed chunks
		self.DIR_PREPROC_CHUNKS = CF.DIR_PREPROC_CHUNKS
		# output directory average semantic vectors
		self.OUT_DIR_SEM_PROPS = CF.DIR_SEM_CHUNK_PROPS

		# check folders
		if not os.path.isdir(self.DIR_PREPROC_CHUNKS):
			os.mkdir(self.DIR_PREPROC_CHUNKS)
		if not os.path.isdir(self.OUT_DIR_SEM_PROPS):
			os.mkdir(self.OUT_DIR_SEM_PROPS)

	def make(self, IN_DIR_PDF, list_files_pdf=[]):
		
		t0 = time()

		listfiles = list_files_pdf or os.listdir(IN_DIR_PDF)

		cpf = 0
		for infilepdf in listfiles:
			cpf += 1
			print("{}/{}".format(cpf, len(listfiles)), flush=True, end="\r")
			
			infiletxt = infilepdf.rsplit(".", 1)[0] + ".txt"
			
			fw = open(self.OUT_DIR_SEM_PROPS+infiletxt, "w", encoding="UTF8")
			
			for l in open(self.DIR_PREPROC_CHUNKS+infiletxt, "r", encoding="UTF8"):
				if not l.rstrip():
					continue

				sp = l.split("\t", 1)

				prewords = sp[1].rstrip().split(" ")

				avgvec = self.WET.AvgEmbedding(prewords)

				# Avg Embedding Vector (sep ",")
				fw.write("{}\n".format(",".join(map(str, avgvec))))

		print("Execution Time: " + str(time() - t0))
