# SDM
Title: Hybrid Method to Automatically Extract Medical Document Tree Structure (2022)

Journal: Engineering Applications of Artificial Intelligence

Authors: Mohamed Yassine Landolsi, Lobna Hlaoua, Lotfi Ben Romdhane

### Setup ###
> 1. execute this console command (modules):
> - pip install -r requirements.txt
> 2. download our pre-trained models?:
> - https://drive.google.com/drive/folders/1jEse714ZF80bntL-YIkgucf_bI969bBT?usp=share_link

### How to train/test? ###
go to "main/" and execute "main.py".

### How to find the results? ###
all generated files are in "utils/data" (e.g. generated trees in "utils/data/INS/outputPredTitles/" folder).

### Customize ###

#### Input ####
1. make a folder like this: "main/data{NAME_DS}/" (e.g. "main/dataINS/")
2. go to "main/dataINS/": put your PDF files in "trainPDF/", "validPDF/" folders.
3. edit "main/conf.py" and change "NAME_DS" parameter (e.g. by "INS").

#### How to change default parameters? ####
- change training/evaluation options in "main/conf.py".
- change model checkpoint directory (DIR_CKPT_SECTDET) in "main/conf_obj.py".
- change model parameters "utils/runCNN1V.py" and other parameters in "main/conf_obj.py".

#### Manual annotation ####
1. put your PDF valid documents (e.g. in "main/dataINS/validPDF/").
2. edit "main/conf.py", change "IS_TRAIN" by 0 and change "EVAL_AUTO" by 1.
3. create your manual annotations files in "manu_annot/" folder (e.g. "main/dataINS/manu_annot/"):
* each line is 1/0 (title/non-title).
* each line represent a chunk, its order must match the one in the generated tree file (e.g. in "utils/data/INS/outputPredTitles/").
