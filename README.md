Here will be described basicactions, that allowed to reproduce results for test case repair for Python dataset for fine-tuned and instruction-tuned models, as well as results for instruction-tuned models for Java dataset to correctly run this program python of a version 3..12 or higher is required.

the most inportant class to do it is eftt(encode, fine-tune, test), whole process was done in jupyter notebooks, here it was shown in blocks that were used, but with little adaptations it can be adapted to any interface.
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

If testing deepseek is desired then it is also necessary to install newest version of transformers directly from directory (as of 16.05.2025):
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
# Add project root to sys.path
project_root = Path().resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
%cd data_processing
# Add project root to sys.path
project_root = Path().resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
%cd CodeBLEU
# Add project root to sys.path
project_root = Path().resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
%cd ..
%cd jCodeBLEU
# Add project root to sys.path
project_root = Path().resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
%cd ..
%cd ..
```
