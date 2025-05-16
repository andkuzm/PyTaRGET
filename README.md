Here will be described basicactions, that allowed to reproduce results for test case repair for Python dataset for fine-tuned and instruction-tuned models, as well as results for instruction-tuned models for Java dataset to correctly run this program python of a version 3.12 or higher is required.

the most inportant class to do it is Eftt(encode, fine-tune, test), whole process was done in jupyter notebooks, here it was shown in blocks that were used, but with little adaptations it can be adapted to any interface.
All code written below was completed in the order it is written, it is assumed that PyTaRGET is cloned 
```
git clone https://github.com/andkuzm/PyTaRGET.git 
```
and then it is necessary to install external packages, for that with the environment, in which the code will be run, it is necessary to move into the PyTaRGET directory (cd PyTaRGET) and run 
```
pip install -r requirements.txt
```

after that it is necessary to install some additional packages:
```
pip install tree-sitter==0.24.0
```
when installing this package it is expected pip to raise a warning, since it is not compatible with some of the functions of other packages.

If testing deepseek is desired then it is also necessary to install newest version of transformers directly from directory (as of 16.05.2025 pip install transformers does not install new enough version):
```
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```

it is assumed before execution of the code, that the present working directory (pwd) is the one in which PyTaRGET folder is located.

First step would allow to correctly import PyTaRGET packages. Most of the IDEs handle that without any additional tinkering, but jupyter notebooks require running this block before anything else:
```
import sys
from pathlib import Path

%cd PyTaRGET
project_root = Path().resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
%cd data_processing
project_root = Path().resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
%cd CodeBLEU
project_root = Path().resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
%cd ..
%cd jCodeBLEU
project_root = Path().resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
%cd ..
%cd ..
```
**Fine-tuned models**
Then this code can be run to create Instance of Eftt class assuming that dataset with annotated cases is located at the root of PyTaRGET directory:

```
from data_processing.encode_tune_test import Eftt
annotated_cases_path = Path("annotated_cases.csv")
out_path = Path("data_processing/results")
model = "codet5p"
train_size = 0.8
beam_size = 5
experiment = Eftt(annotated_cases_path, out_path, model, train_size, beam_size, hftoken=None, batch_size=1, java=False)
```
(Possible values for the model parameter out of fine-tuned models are: codet5p, plbart, codegen)
this way Eftt class will be created, in this instance it can be used to encode the prepare and encode dataset for codet5p model, in which training and validation sets will have size of 80% of the whole dataset, and test dataset will be of the size of remaining 20%, datasets will be created in a directory in PyTaRGET/data_processing/results/0.8. and to do it it is necessary to run:```experiment.encode()``` command.

After the datasets are ready, fine tuning can be performed for the modelduring fine-tuning batch_size parameter will be used to determine size of batches, higher value will speed up the process but require more video random access memory (VRAM). This process can be started using: ```experiment.train()``` command.

Finally to test the model ```experiment.validate()``` can be used. it will take checkpoint of the models created during training, make predictions with it on the test set, and then get BLEU, CodeBLEU and Exact Match metrics for the predictions. For validation of fine-tuned models beam_size parameter will be used, it determines how many predictions will be made for every target, higher values require more VRAM.

**Instruction-tuned models**
To test Instruction-tuned models it is again necessary to create Eftt instance first:
```
from data_processing.encode_tune_test import Eftt
annotated_cases_path = Path("annotated_cases.csv")
out_path = Path("data_processing/results")
model = "qwen"
train_size = 0.8
beam_size = 5
experiment = Eftt(annotated_cases_path, out_path, model, train_size, beam_size, hftoken=None, batch_size=1, java=False)
```
