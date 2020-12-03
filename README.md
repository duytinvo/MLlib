# MLlib

## Organization
This library is used to construct different NN architectures using pytorch that could be applied to solve classification, tagger and seq2seq problems. The core package is lain on the ``mlmodels`` folder, which consists of 06 sub-folders

### ``utils`` folder
This folder contains scripts to:
 1. Read input data from varied formats: **.txt**, **.json**, **.csv**
 2. Process raw text by applying tokenization techniques: nltk, wordpiece
 3. Either build or map vocabulary
 4. Convert text lists to index lists 
 5. Transform index lists to tensor inputs feeding to a NN model
 
### ``modules`` folder
This folder is employed for building:

| Core Nets | Complex Nets | Transformer | End-to-End Nets |
| --- | --- | --- | --- |
| Embedded layer  | Encoder  | Self-attention | Seq2seq |
| (Bi-)CNN layer  | Decoder  | BERT | Labeler |
| (Bi-)RNN layer | CRF  | CTRL | Classifier |
| (Bi-)GRU layer | GraphNN  | T5 | --- |
| (Bi-)LSTM layer  | ---  | --- | --- |
| Copy and Pointer  | ---  | --- | --- |
| Attention layer  | ---  | --- | --- |
| --- | --- | --- | --- |

### ``metrics`` folder
This folder consists of evaluation methods using different metrics such as Precision, Recall, F1-score, Blue score, Ranking score, Matching score, ...
### ``training`` folder
This folder involves a fully connected model called from the ``modules`` folder that is ready for training and inference.  
### ``inference`` folder
This folder provides serving functions for models trained from the ``Builder`` folder
### ``demo`` folder
This folder contain all training and inference script used for deployment in leaf repos   

## Contribution Procedure
When working with a ML problem P given a training set D to estimate output labels/values
### Use case 1: the same model architecture when the task is similar with other supported task for example NER (D) and ABSA (D')
- Directly use NNlib (partially contribute)
    - Create a leaf repo
    - Copy all relevant file in the ``demo`` folder
    - Reformat D' data to follow D format/ Write a new inputIO in the ``utils`` folder (Create new branch and submit PR for code review if contributing back to NNlib)
    - Train, deploy and test
### Use case 2: different model architecture when the task is similar with other supported task for example NER (D) and ABSA (D')
- Indirectly use NNlib (partially contribute)
    - Create a new architecture in the ``modules`` folder (Create new branch and submit PR for code review if contributing back to NNlib)
    - Create a leaf repo
    - Copy all relevant file in the ``demo`` folder
    - Reformat D' data to follow D format/ Write a new inputIO in the ``utils`` folder (Create new branch and submit PR for code review if contributing back to NNlib)
    - Train, deploy and test
### Use case 3: the task is unsupported (e.g. classifier)
- Created a new sub-library (fully contribute)
    - On the NNlib repo:
        - Create a new branch
        - Write new **inputIO.py** in the ``utils`` folder **(if unsupported yet)**
        - Write new **vocab_builder.py** in the ``utils`` folder **(if unsupported yet)**
        - Write new core nets (**corenet.py**) in the ``modules`` folder **(if unsupported yet)** 
        - Write new **classifier.py** in the ``modules`` folder by connecting core nets together
        - Write new **metric.py** in the ``metrics`` folder **(if unsupported yet)**
        - Write new **classifier_model.py** in the ``training`` folder
        - Write new **classifier_serve.py** in the ``inference`` folder
        - Write new deployment files (**classifier_settings.py**, **classifier_inference.py**, **classifier_train.py**) in the ``demno`` folder
        - Run and test on this branch
        - Submit PR for code review
    - Create a leaf repo
        - Copy all relevant files in the ``demo`` folder
        - Train, deploy and test
    