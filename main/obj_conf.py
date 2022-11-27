import sys
sys.path.append("../main")
from conf import *
sys.path.append(DIR_SECTDET)

class CF():
	def __init__(self, NAME_DS):

		""" Input """

		# - documents type
		self.NAME_DS = NAME_DS

		# - directories/paths
		self.DIR_DATA_MAIN = f"{DIR_MAIN}data{self.NAME_DS}/"
		self.DIR_TRAIN_PDF = f"{self.DIR_DATA_MAIN}trainPDF/"
		self.DIR_VALID_PDF = f"{self.DIR_DATA_MAIN}validPDF/"
		self.DIR_MANU_ANNOT = f"{self.DIR_DATA_MAIN}manu_annot/"

		""" Output Directories """

		# Chunks data
		self.DIR_DATA_SECTDET = f"{DIR_SECTDET}data/{self.NAME_DS}/"
		self.DIR_FORM_CHUNK_PROPS = f"{self.DIR_DATA_SECTDET}outputFProps/"
		self.DIR_SEM_CHUNK_PROPS = f"{self.DIR_DATA_SECTDET}outputSProps/"
		self.DIR_PREPROC_CHUNKS = f"{self.DIR_DATA_SECTDET}outputPreTxt/"
		self.DIR_AUTO_CHUNKS = f"{self.DIR_DATA_SECTDET}outputChunks/"
		self.DIR_FORM_WORD = f"{self.DIR_DATA_SECTDET}outputForm/"
		self.DIR_TOP_STYLES = f"{self.DIR_DATA_SECTDET}outputStyles/"
		self.DIR_CHUNK_POSITIONS = f"{self.DIR_DATA_SECTDET}outputPosi/"

		# Models directory
		self.DIR_MODELS = f"{DIR_SECTDET}models/{self.NAME_DS}/"

		# Section detection model
		self.DIR_PREDS_SECTDET = f"{self.DIR_DATA_SECTDET}outputPredTitles/"
		self.DIR_CKPT_SECTDET = f"{self.DIR_MODELS}modelcnn1d_auto3031/" # modelcnn1d_manual320
		self.FILE_CKPT_SECTDET = "modelTitles.h5"

		# Word Fasttext embedding
		self.PATH_EMB_CKPT = f"{self.DIR_MODELS}fasttext/model.bin"

		""" Parameters """

		# Chunking parameters:
		# - min percent of capitalized chars for a capitalized chunk
		self.MIN_PERC_UPPERCASE = 0.85

		# Chunking parameters default language (EN):
		# - K top fonts that can be titles
		self.K_TOP_FONTS = 3
		# - max words per title
		self.MAX_NB_WORDS_TITLE = 20
		# - chunk starts with these words can't be a title
		self.IGNORE_TITLES = {"remarque", "note", "warning", "alert", "table", "figure", "diagram", "fig", "tab"}
		# - header/footer heights
		if self.NAME_DS=="SmPC":
			self.HEAD_FOOT_HEIGHT = None
		else:
			self.HEAD_FOOT_HEIGHT = [62, 55]
