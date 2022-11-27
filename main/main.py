import sys
sys.path.append("../main")
from conf import *
from obj_conf import CF
sys.path.append(DIR_SECTDET)
from WordEmbTools import WordEmbTools
from PreTextChunks import PreTextChunks
from WordEmbTrain import WordEmbTrain
from CalcSemaProps import CalcSemaProps
from LearnCorrectTitles import LearnCorrectTitles
import dill
dill.load_session(f"{DIR_SECTDET}bin/CutPrepareChunks.bin")
dill.load_session(f"{DIR_SECTDET}bin/PDFread.bin")

# - parameters object according to data type
CF = CF(NAME_DS)

""" ---- CONSTRUCT TRAIN & TEST SET ---- """

print("------ NER {} ------".format(NAME_DS))

# Data set modes
if not IS_TRAIN:
	# - construct test data
	modes = [0]
else:
	# - construct train data & test data
	modes = [1, 0]

# Iterate modes
for IS_TRAIN_DS in modes:

	""" ------------------------ """

	if IS_TRAIN_DS:
		# PDF directory
		IN_DIR_PDF = CF.DIR_TRAIN_PDF
	else:
		IN_DIR_PDF = CF.DIR_VALID_PDF

	""" ------------------------ """

	print("------ {} set ------".format(["Valid", "Train"][IS_TRAIN_DS]))

	""" prepare NLP tools """
	print("###### Prepare NLP tools ######")
	WET = WordEmbTools(CF)

	""" parse PDF """
	print("###### Parse PDF ######")
	PDFread = PDFread(CF)
	PDFread.make(IN_DIR_PDF)

	""" cut & prepare chunks """
	print("###### Cut & prepare chunks ######")
	CutPrepareChunks = CutPrepareChunks(CF)
	CutPrepareChunks.make(IN_DIR_PDF)

	""" Preprocessing chunks text for section detection """
	print("###### Preprocessing chunks text for section detection ######")
	obj_PreTextChunks = PreTextChunks(CF, WET)
	obj_PreTextChunks.make(IN_DIR_PDF)

	""" Train Fasttext embedding? """
	if IS_TRAIN_DS:
		print("###### Train Fasttext embedding ######")
		obj_WordEmbTrain = WordEmbTrain(CF, WET)
		obj_WordEmbTrain.train(IN_DIR_PDF)

	""" Load trained Fasttext embedding model """
	print("###### Load trained Fasttext embedding model ######")
	WET.loadModel()

	""" Prepare semantic features for section detection """
	print("###### Prepare semantic features for section detection ######")
	obj_CalcSemaProps = CalcSemaProps(CF, WET)
	obj_CalcSemaProps.make(IN_DIR_PDF)

""" Machine learning to annotate chunks by title/non title """
print("###### Machine learning to annotate chunks by title/non title ######")
obj_LearnCorrectTitles = LearnCorrectTitles(CF, WET)
if IS_TRAIN:
	# Train
	obj_LearnCorrectTitles.load(CF.DIR_TRAIN_PDF)
	obj_LearnCorrectTitles.train()
# Evaluate
obj_LearnCorrectTitles.load(CF.DIR_VALID_PDF)
obj_LearnCorrectTitles.detect()
