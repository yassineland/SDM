# SDM
Hybrid Method to Automatically Extract Medical Document Tree Structure

### Setup ###
> 1. execute this console command (modules):
> - pip install -r requirements.txt

### How to train/test? ###
go to "main/" and execute "main.py".

### How to find the results? ###
all generated files are in "utils/data" (e.g. generated trees in "utils/data/INS/outputPredTitles/" folder).

### Customize ###

#### Input ####
1. make a folder like this: "main/data{NAME_DS}/" (e.g. "main/dataINS/")
2. go to "main/dataINS/": put your PDF files in "trainPDF/", "validPDF/" folders.
3. edit "main/conf.py" and change "NAME_DS" parameter.

#### How to change default parameters? ####
- change training/evaluation options in "main/conf.py".
- change model checkpoint directory (DIR_CKPT_SECTDET) in "main/conf_obj.py".
- change model parameters "utils/runCNN1V.py" and other parameters in "main/conf_obj.py".

#### Manual annotation ####
1. put your PDF valid documents (e.g. in "main/dataINS/validPDF/").
2. edit "main/conf.py", change "IS_TRAIN" to 0, change "EVAL_AUTO" to 1.
3. create your manual annotations files in "manu_annot" folder (e.g. "main/dataINS/manu_annot/"):
* each line is 1/0 (title/non-title).
* each line represent a chunk, its order should be the same in the generated tree file (e.g. in "utils/data/INS/outputPredTitles/").
