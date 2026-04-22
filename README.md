In this section will be described actions, that would allow to reproduce results for test case repair for Python dataset for fine-tuned and instruction-tuned models, as well as results for instruction-tuned models for Java dataset to correctly run this program python of a version 3.12 or higher is required.

the most important class to do it is Eftt(encode, fine-tune, test), whole process was done in jupyter notebooks, here it was shown in blocks that were used, but with little adaptations it can be adapted to any interface.
All code written below was completed in the order it is written, it is assumed that PyTaRGET is cloned 
```
git clone https://github.com/andkuzm/PyTaRGET.git 
```
and then it is necessary to install external packages, for that with the environment, in which the code will be run, it is necessary to move into the PyTaRGET directory (cd PyTaRGET) and run 
```
pip install -r requirements.txt
```

after that it is necessary to install an additional package:
```
pip install tree-sitter==0.24.0
```
when installing this package it is expected pip to raise a warning, since it is not compatible with some of the functions of other packages.

If testing deepseek is desired then it is also necessary to install newest version of transformers directly from directory (as of 16.05.2025 pip install transformers does not install new enough version):
```
git clone https://github.com/HuggingFace/transformers.git
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
beam_size = 20
experiment = Eftt(annotated_cases_path, out_path, model, train_size, beam_size, hftoken=None, batch_size=1, java=False)
```
(Possible values for the model parameter out of fine-tuned models are: codet5p, plbart, codegen)
this way Eftt class will be created, in this instance it can be used to encode the prepare and encode dataset for codet5p model, in which training and validation sets will have size of 80% of the whole dataset, and test dataset will be of the size of remaining 20%, datasets will be created in a directory in PyTaRGET/data_processing/results/0.8. and to do it it is necessary to run:```experiment.encode()``` command.

After the datasets are ready, fine tuning can be performed for the modelduring fine-tuning batch_size parameter will be used to determine size of batches, higher value will speed up the process but require more video random access memory (VRAM). This process can be started using: ```experiment.train()``` command.

Finally to test the model ```experiment.validate()``` can be used. it will take checkpoint of the models created during training, make predictions with it on the test set, and then get BLEU, CodeBLEU and Exact Match metrics for the predictions. For validation of fine-tuned models beam_size parameter will be used, it determines how many predictions will be made for every target, higher values require more VRAM.

If it is required to get metrics from the predictions make this can be used: ```experiment.get_metrics()```, under the assumption that the file with predictions the model made is not moved, nor renamed.

**Instruction-tuned models**

To test Instruction-tuned models it is again necessary to create Eftt instance first:
```
from data_processing.encode_tune_test import Eftt
annotated_cases_path = Path("annotated_cases.csv")
out_path = Path("data_processing/results")
model = "qwen"
train_size = 0.0
beam_size = 5
experiment = Eftt(annotated_cases_path, out_path, model, train_size, beam_size, hftoken=None, batch_size=1, java=False)
```
(Possible values for the model parameter out of fine-tuned models are: qwen, qwen3, deepseek and gemma)
Here HuggingFace is required to be passed as hftoken parameter, for gemma token should belong to an account with access to it.
Encoding once again should be done with ```experiment.encode()```, which will again create three datasets, but the only one important for instruction-tuned models is test dataset, so train_fraction parameter can simple be used to determine proportion of the dataset that would be used for testing (1-train_fraction).

After encoding is done testing can be started, for that ```experiment.validate_llm()``` should be used, it will load chosen model, tokenize inputs, insert instructions and pass it to the model, then it will append predictions it gave to the results file, fine-tuned models create file with instructions only after the whole dataset is tested, instruction-tuned models append them as they are produced, so validation can be divided into several sessions.

If it is required to get metrics from the predictions make this can be used: ```experiment.get_metrics_llm()```, under the assumption that the file with predictions the model made is not moved, nor renamed.

**Java Instruction-tuned models**

To test Instruction-tuned models on java dataset first it is necessary to have any of the .json files, that were created after encoding process  was completed as it is described in this project: https://github.com/Ahmadreza-SY/TaRGET/tree/master?tab=readme-ov-file#test-case-repair-data-collection

After the file is created. it should be brought into the form that is necessary for our project, by creating any instance of the Eftt class and running ```eftt_instance.reannotate("Path/to/annotated_java_dataset.json")``` this would create reannotated file named test.json in data_processing/results/model_name/train_fraction folder. this file then should be moved into splits folder, so in the end its location from root folder should be data_processing/results/model_name/train_fraction/splits/test.json

now that we have the annotated java dataset, we can move to evaluation of the Instruction-tuned LLM for this dataset. But before it, right now the program determines whether the passed dataset is that in java language or python by the name of train_fraction parameter, if testing with java code, train_fraction parameter passed with Eftt creation should be "ref" for prompt to be correct, so actual location of the java dataset file should be data_processing/results/model_name/ref/splits/test.json. If for any reason other name is desired, it can be changed by changing if statement in the llm_test.py file on the 48th line to work functionally and also on the 32th line for print about java prompt being used to work correctly.
Then to test The model it is necessary to just run it the same way that it was previously done, i.e.:
```
from data_processing.encode_tune_test import Eftt
annotated_cases_path = Path("annotated_cases.csv")
out_path = Path("data_processing/results")
model = "qwen"
train_size = "ref"
beam_size = 5
experiment = Eftt(annotated_cases_path, out_path, model, train_size, beam_size, hftoken="HuggingFace token", batch_size=1, java=False)
experiment.validate_llm()
```
for qwen model

But codebleu metrics will not work correctly for java target-to-prediction, to solve this problem the way that was used in the referenced project was used for this one, but it requires additional actions, for that - after the desired number of predictions is made it is necessary to:
1. Create a virtual environment as it was instructed in https://github.com/Ahmadreza-SY/TaRGET/tree/master?tab=readme-ov-file#test-case-repair-data-collection, or use the existing one, if Java dataset encoding process was completed with it.
2. Install all packages into that environment as it was instructed in the referenced project if new environment was created during first step.
3. With this environment, again create the same Eftt, that was used to make predictions but with java parameter being True, i.e.:
```
from data_processing.encode_tune_test import Eftt
annotated_cases_path = Path("annotated_cases.csv")
out_path = Path("data_processing/results")
model = "qwen"
train_size = "ref"
beam_size = 5
experiment = Eftt(annotated_cases_path, out_path, model, train_size, beam_size, hftoken="HuggingFace token", batch_size=1, java=True)
```
4. Run with this instance ```experiment.get_metrics_llm()```, when doing so, CodeBLEU metric will take specifics of Java language into account. 

**Reproduction**

To reproduce the processes using methods described previously, results.zip file that is located in data_processing folder should be extracted into the same folder, afterwards one can run fine-tuning on models suitable for it right away on 0.4, 0.6 or 0.8 train_fractions and afterwards run validation process with the methods that vere previously described. to get exactly the same results, it is necessary to use beam_size=20, and epochs and early early stop necessary are hard coded into the train.py file. encoding process can also be completed, but if encoding process is done, by default test dataset will have the size of 1-train_fraction, while the preuploaded results have it uniform 0.2, smallest out of tested, so unless it is manually swapped to 0.2 test.json file, only when passing 0.8 as train_fraction parameter will the same exact results be received.

For instruction-tuned models, there is a ref results, where an already reannotated and reduced test.json file for Java dataset is located, so for ref only ```.validate_llm()``` should be done, and the only other train_fraction used is 0.0, which conversively means test.json is the whole Python dataset. Predictions with which the metrics referred to in the paper were received are in data_processing/results/model_name/train_fraction/model_name_llm_test_predictions.json file. It is important to notice that instruction-tuned LLMs do not have random_seed parameter, or its analog, and thus when generating predictions receiving the same results is highly unlikely, but the ones that are present in the compressed results can be validated, and evaluated with ```.get_metrics_llm()``` ,method as described previously.

**Python dataset collection**

Dataset collection is done by running GitHubSearch.py directly from any directory:
```
python repository_search/GitHubSearch.py
```
It will ask for a GitHub token (needs public read access) and a version name. The version name is used to name all output files, for example with version name "v1":
- `annotated_cases_v1.csv` — the collected dataset, written to the current directory
- `processed_repositories_v1.txt` — list of already-processed repositories, also in the current directory
- `repos_v1/` — temporary directory where repositories are cloned during processing

If those files already exist the run resumes from where it left off, skipping repositories that were already processed.

Right now the tool will search through up to a thousand repositories per query (there are 3 queries for different licenses), sorted by stars, so up to 3000 total. If it is desired to process more, the easiest way is to set tighter size boundaries by passing smaller `size_start`/`size_end` values to `find_and_process_repositories` and running multiple passes. The parameters and their defaults are in `GitHubSearch.py` on line 197.

The `processed_repositories_{version}.txt` file acts as a blacklist: entries in the format `repository_owner/repository_name|latest_commit_hash` prevent the same repository from being processed twice. It also guards against repositories that could break installed packages — to run pytest inside each repository a subprocess is used, and to resolve imports in that subprocess every repository is installed with `pip install -e .`, which can occasionally corrupt packages in the outer environment. Repositories known to cause this can be added to the file manually.
