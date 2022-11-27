import re
import os
from time import time

#####################################

class PreTextChunks(object):

	def __init__(self, CF, WET):

		# re-preprocess already existing preprocessed documents
		self.OVERWRITE = True

		# embedding & NLP tools object
		self.WET = WET

		# input directory of chunks
		self.DIR_AUTO_CHUNKS = CF.DIR_AUTO_CHUNKS
		# output directory average semantic vectors
		self.OUT_DIR_PREPROC_CHUNKS = CF.DIR_PREPROC_CHUNKS

		# prepare directories
		if not os.path.isdir(CF.DIR_AUTO_CHUNKS):
			os.mkdir(CF.DIR_AUTO_CHUNKS)
		if not os.path.isdir(self.OUT_DIR_PREPROC_CHUNKS):
			os.mkdir(self.OUT_DIR_PREPROC_CHUNKS)

	def make(self, IN_DIR_PDF, list_files_pdf=[]):
		
		t0 = time()

		listfiles = list_files_pdf or os.listdir(IN_DIR_PDF)

		cpf = 0
		for infilepdf in listfiles:
			cpf += 1
			print("{}/{}".format(cpf, len(listfiles)), flush=True, end="\r")
			
			infiletxt = infilepdf.rsplit(".", 1)[0] + ".txt"

			if not self.OVERWRITE and os.path.isfile(self.OUT_DIR_PREPROC_CHUNKS+infiletxt):
				continue

			fw = open(self.OUT_DIR_PREPROC_CHUNKS+infiletxt, "w", encoding="UTF8")
			
			for l in open(self.DIR_AUTO_CHUNKS+infiletxt, "r", encoding="UTF8"):

				sp = l.rstrip().split(". ", 1)
				
				chunk = sp[1]

				prewords, poss = self.WET.preprocTxt(chunk, pos=1, empty=1)

				# PoS (sep ",") <TAB> preprocessed words (sep " ")
				fw.write("{}\t{}\n".format(",".join(map(str, poss)), " ".join(prewords)))

		print("Execution Time: " + str(time() - t0))
