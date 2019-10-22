# NLP Toolkit
Library containing state-of-the-art models for Natural Language Processing tasks  
The purpose of this toolkit is to allow for **easy training/inference of state-of-the-art models**, for various NLP tasks.  
*Note this repo currently in development. (see [To do list](#to-do-list)) 

---

## Contents
**Tasks**:  
1. [Classification](#1-classification)
2. [Automatic Speech Recognition](#2-automatic-speech-recognition)
3. [Text Summarization](#3-text-summarization)
4. [Machine Translation](#4-machine-translation)
5. [Natural Language Generation](#5-natural-language-generation)
6. [Punctuation Restoration](#6-punctuation-restoration)  
7. [Named Entity Recognition](#7-named-entity-recognition)
  
[Benchmark Results](#benchmark-results)  
[References](#references)

---

## Pre-requisites
torch==1.2.0 ; spacy==2.1.8 ; torchtext==0.4.0 ; seqeval==0.0.12 ; pytorch-nlp==0.4.1  
For mixed precision training (-fp16=1), apex must be installed: [apex==0.1](https://github.com/NVIDIA/apex)  
For chinese support in Translation: jieba==0.39  
For ASR: librosa==0.7.0 ; soundfile==0.10.2  
For more details, see requirements.txt

** Pre-trained models (XLNet, BERT, GPT-2) are courtesy of huggingface (https://github.com/huggingface/pytorch-transformers)

## Package Installation
```bash
git clone https://github.com/plkmo/NLP_Toolkit.git
cd NLP_Toolkit
pip install .

# to uninstall if required to re-install after updates,
# since this repo is still currently in active development
pip uninstall nlptoolkit 
```
Alternatively, you can just use it as a non-packaged repo after git clone.

---

## 1) Classification
The goal of classification is to segregate documents into appropriate classes based on their text content. Currently, the classification toolkit uses the following models:
1. Text-based Graph Convolution Networks (GCN)
2. Bidirectional Encoder Representations from Transformers (BERT)
3. XLNet

### Format of datasets files
The training data (default: train.csv) should be formatted into two columns 'text' and 'label' respectively, with rows being the documents index. 'text' contains the raw text and 'label' contains the corresponding label (integers 0, 1, 2... depending on the number of classes)

The infer data (default: infer.csv) should be formatted into at least one column 'text' being the raw text and rows being the documents index. Optional column 'label' can be added and --train_test_split argument set to 1 to use infer.csv as the test set for model verification.

- IMDB datasets for sentiment classification available [here.](https://drive.google.com/drive/folders/1a4tw3UsbwQViIgw08kwWn0jvtLOSnKZb?usp=sharing)

### Running the model
Run classify.py with arguments below.

```bash
classify.py [-h] 
	[--train_data TRAIN_DATA (default: "./data/train.csv")] 
	[--infer_data INFER_DATA (default: "./data/infer.csv")]            
	[--max_vocab_len MAX_VOCAB_LEN (default: 7000)]  
	[--hidden_size_1 HIDDEN_SIZE_1 (default: 330)]
	[--hidden_size_2 HIDDEN_SIZE_2 (default: 130)]
	[--tokens_length TOKENS_LENGTH (default: 200)] 
	[--num_classes NUM_CLASSES (default: 2)]
	[--train_test_split TRAIN_TEST_SPLIT (default: 0)]
	[--test_ratio TEST_RATIO (default: 0.1)] 
	[--batch_size BATCH_SIZE (default: 32)]      
	[--gradient_acc_steps GRADIENT_ACC_STEPS (default: 1)]
	[--max_norm MAX_NORM (default: 1)] 
	[--num_epochs NUM_EPOCHS (default: 1700)] 
	[--lr LR default=0.0031]
	[--model_no MODEL_NO (default: 0 (0: Graph Convolution Network 	(GCN), 1: BERT, 2: XLNet))] 
	[--train TRAIN (default:1)]  
	[--infer INFER (default: 0 (Infer input sentence labels from 	trained model))]
```
The script outputs a results.csv file containing the indexes of the documents in infer.csv and their corresponding predicted labels.

Or if used as a package:
```python
from nlptoolkit.utils.config import Config
from nlptoolkit.classification.models.BERT.trainer import train_and_fit
from nlptoolkit.classification.models.infer import infer_from_trained

config = Config(task='classification') # loads default argument parameters as above
config.train_data = './data/train.csv' # sets training data path
config.infer_data = './data/infer.csv' # sets infer data path
config.num_classes = 2 # sets number of prediction classes
config.batch_size = 32
config.model_no = 1 # sets BERT model
config.lr = 0.001 # change learning rate
train_and_fit(config) # starts training with configured parameters
inferer = infer_from_trained(config) # initiate infer object, which loads the model for inference, after training model
inferer.infer_from_input() # infer from user console input
inferer.infer_from_file(in_file="./data/input.txt", out_file="./data/output.txt")
```

```python
inferer.infer_from_input()
```
Sample output:
```bash
Type input sentence (Type 'exit' or 'quit' to quit):
This is a good movie.
Predicted class: 1

Type input sentence (Type 'exit' or 'quit' to quit):
This is a bad movie.
Predicted class: 0

```

### Pre-trained models
Download and zip contents of downloaded folder into ./data/ folder.
1. [BERT for IMDB sentiment analysis](https://drive.google.com/drive/folders/1JHOabZE4U4sfcnttIQsHcu9XNEEJwk0X?usp=sharing) (includes preprocessed data, vocab, and saved results files)
2. [XLNet for IMDB sentiment analysis](https://drive.google.com/drive/folders/1lk0N6DdgeEoVhoaCrC0vysL7GBJe9sAX?usp=sharing) (includes preprocessed data, vocab, and saved results files)
---

## 2) Automatic Speech Recognition
Automatic Speech Recognition (ASR) aims to convert audio signals into text. This library contains the following models for ASR: 
1. Speech-Transformer
2. Listen-Attend-Spell (LAS)

### Format of dataset files
The folder containing the dataset should have the following structure: folder/speaker/chapter
Within the chapter subdirectory, the audio files (in .flac format) are named speaker-chapter-file_id (file_id In running order)
The transcript .txt file for the files within the chapter should be located in the chapter subdirectory. In the transcript file, each row should consist of the speaker-chapter-file_id (space) transcript.

### Running the model
Run speech.py with arguments below

```bash
speech.py [-h] 
	[--folder FOLDER (default: train-clean-5")] 
	[--level LEVEL (default: word")]   
	[--use_lg_mels USE_LG_MELS (default: 1)]
	[--use_conv USE_CONV (default: 1)]
	[--n_mels N_MELS (default: 80)]
	[--n_mfcc N_MFCC (default: 13)]
	[--n_fft N_FFT (default: 25)]
	[--hop_length HOP_LENGTH (default: 10)]
	[--max_frame_len MAX_FRAME_LEN (default: 1000)]
	[--d_model D_MODEL (default: 64)]
	[--ff_dim FF_DIM (default: 128)]
	[--num NUM (default: 6)]
	[--n_heads N_HEADS(default: 4)]
	[--batch_size BATCH_SIZE (default: 30)]
	[--fp16 FP16 (default:1)]  
	[--num_epochs NUM_EPOCHS (default: 8000)] 
	[--lr LR default=0.003]    
	[--gradient_acc_steps GRADIENT_ACC_STEPS (default: 4)]
	[--max_norm MAX_NORM (default: 1)] 
	[--T_max T_MAX (default: 5000)]  
	[--model_no MODEL_NO (default: 0 (0: Transformer, 1: LAS))]  
	[--train TRAIN (default:1)]  
	[--infer INFER (default: 0 (Infer input sentence labels from 	trained model))]

```

---

## 3) Text Summarization
Text summarization aims to distil a paragraph chunk into a few sentences that capture the essential information. This library contains the following models for text summarization: 
1. Convolutional Transformer 
2. Seq2Seq (LAS architecture)

### Format of dataset files
One .csv file for each text/summary pair. Within the text/summary .csv file, text is followed by summary, with summary points annotated by @highlights (summary) 

### Running the model
Run summarize.py with arguments below

```bash
summarize.py [-h] 
	[--data_path DATA_PATH] 
	[--level LEVEL (default: bpe")]   
	[--bpe_word_ratio BPE_WORD_RATIO (default: 0.7)]
	[--bpe_vocab_size BPE_VOCAB_SIZE (default: 7000)]
	[--max_features_length MAX_FEATURES_LENGTH (default: 200)]
	[--d_model D_MODEL (default: 128)]
	[--ff_dim FF_DIM (default: 128)]
	[--num NUM (default: 6)]
	[--n_heads N_HEADS(default: 4)]
	[--LAS_embed_dim LAS_EMBED_DIM (default: 128)]
	[--LAS_hidden_size LAS_HIDDEN_SIZE (default: 128)]
	[--batch_size BATCH_SIZE (default: 32)]  
	[--fp16 FP16 (default: 1)]  
	[--num_epochs NUM_EPOCHS (default: 8000)] 
	[--lr LR default=0.003]    
	[--gradient_acc_steps GRADIENT_ACC_STEPS (default: 4)]
	[--max_norm MAX_NORM (default: 1)] 
	[--T_max T_MAX (default: 5000)]  
	[--model_no MODEL_NO (default: 0 (0: Transformer, 1: LAS))]  
	[--train TRAIN (default:1)]  
	[--infer INFER (default: 0 (Infer input sentence labels from 	trained model))]

```

Or if used as a package:
```python
from nlptoolkit.utils.config import Config
from nlptoolkit.summarization.trainer import train_and_fit
from nlptoolkit.summarization.infer import infer_from_trained

config = Config(task='summarization') # loads default argument parameters as above
config.data_path = "./data/cnn_stories/cnn/stories/"
config.batch_size = 32
config.lr = 0.0001 # change learning rate
config.model_no = 0 # set model as Transformer
train_and_fit(config) # starts training with configured parameters
inferer = infer_from_trained(config) # initiate infer object, which loads the model for inference, after training model
inferer.infer_from_input() # infer from user console input
inferer.infer_from_file(in_file="./data/input.txt", out_file="./data/output.txt")
```

---

## 4) Machine Translation
The goal of machine translation is to translate text from one form of language to another. This library contains the following models to accomplish this:
1. Transformer

Currently supports translation between: English (en), French (fr), Chinese (zh)

### Format of dataset files
A source .txt file with each line containing the text/sentence to be translated, and a target .txt file with each line containing the corresponding translated text/sentence

### Running the model
Run translate.py with arguments below

```bash
translate.py [-h]  
	[--src_path SRC_PATH]
	[--trg_path TRG_PATH] 
	[--src_lang SRC_LANG (en, fr, zh)] 
	[--trg_lang TRG_LANG (en, fr, zh)] 
	[--batch_size BATCH_SIZE (default: 50)]
	[--d_model D_MODEL (default: 512)]
	[--ff_dim FF_DIM (default: 2048)]
	[--num NUM (default: 6)]
	[--n_heads N_HEADS(default: 8)]
	[--max_encoder_len MAX_ENCODER_LEN (default: 80)]
	[--max_decoder_len MAX_DECODER_LEN (default: 80)]	
	[--fp16 FP_16 (default: 1)]
	[--num_epochs NUM_EPOCHS (default: 500)] 
	[--lr LR default=0.0001]    
	[--gradient_acc_steps GRADIENT_ACC_STEPS (default: 1)]
	[--max_norm MAX_NORM (default: 1)] 
	[--T_max T_MAX (default: 5000)] 
	[--model_no MODEL_NO (default: 0 (0: Transformer))]  
	[--train TRAIN (default:1)]  
	[--evaluate EVALUATE (default:0)]
	[--infer INFER (default: 0)]
	
```

Or if used as a package:
```python
from nlptoolkit.utils.config import Config
from nlptoolkit.translation.trainer import train_and_fit
from nlptoolkit.translation.infer import infer_from_trained

config = Config(task='translation') # loads default argument parameters as above
config.src_path = './data/translation/eng_zh/news-commentary-v13.zh-en.en' # sets source language data path
config.trg_path = './data/translation/eng_zh/news-commentary-v13.zh-en.zh' # sets target language data path
config.src_lang = 'en' # sets source language
config.trg_lang = 'zh' # sets target language
config.batch_size = 16
config.lr = 0.0001 # change learning rate
train_and_fit(config) # starts training with configured parameters
inferer = infer_from_trained(config) # initiate infer object, which loads the model for inference, after training model
inferer.infer_from_input() # infer from user console input
inferer.infer_from_file(in_file="./data/input.txt", out_file="./data/output.txt")
```

---

## 5) Natural Language Generation
Natural Language generation (NLG) aims to generate text based on past context. For instance, a chatbot can generate text replies based on the context of chat history. We currently have the following models for NLG:
1. Generative Pre-trained Transformer 2 (GPT 2)
2. Conditional Transformer Language Model (CTRL)

### Format of dataset files
1. Generate free text from GPT 2 pre-trained models
2. Generate conditional free text from CTRL pre-trained model

### Running the model
Run generate.py

```bash
generate.py [-h]  
	[--model_no MODEL_NO (0: GPT 2 ; 1: CTRL)]
```

Or if used as a package:
```python
from nlptoolkit.utils.config import Config
from nlptoolkit.generation.infer import infer_from_trained

config = Config(task='generation') # loads default argument parameters as above
config.model_no = 0 # sets model to GPT 2
inferer = infer_from_trained(config, tokens_len=100, top_k_beam=3)
inferer.infer_from_input() # infer from user console input
inferer.infer_from_file(in_file="./data/input.txt", out_file="./data/output.txt")
```

```python
inferer.infer_from_input()
```
Sample output:
```bash
Type your input sentence: 
I would
10/22/2019 02:46:16 PM [INFO]: Generating...
 have liked it to have a more "classic-style" feel to it and be more "real-world" looking, but it was just so fun and I loved the idea of having the characters interact in a real way with each others' faces and voices, and the fact it was all done with a lot less CGI than most anime, I thought it was a pretty great addition and I hope it continues to get more attention from the anime industry. I think the anime community is starting to see

```
---

## 6) Punctuation Restoration
Given unpunctuated (and perhaps un-capitalized) text, punctuation restoration aims to restore the punctuation of the text for easier readability. Applications include punctuating raw transcripts from audio speech data etc. Currently supports the following models:
1. Transformer (PuncTransformer)
2. Bi-LSTM with attention (PuncLSTM)

### Format of dataset files
Currently only supports TED talk transcripts format, whereby punctuated text is annotated by \<transcripts\> tags. Eg. \<transcript\> "punctuated text" \</transcript\>. The "punctuated text" is preprocessed and then used for training.

- TED talks dataset can be downloaded [here.](https://drive.google.com/file/d/1fJpl-fF5bcAKbtZbTygipUSZYyJdYU11/view?usp=sharing)

### Running the model
Run punctuate.py

```bash
punctuate.py [-h] 
	[--data_path DATA_PATH] 
	[--level LEVEL (default: bpe")]   
	[--bpe_word_ratio BPE_WORD_RATIO (default: 0.7)]
	[--bpe_vocab_size BPE_VOCAB_SIZE (default: 7000)]
	[--batch_size BATCH_SIZE (default: 32)]
	[--d_model D_MODEL (default: 512)]
	[--ff_dim FF_DIM (default: 2048)]
	[--num NUM (default: 6)]
	[--n_heads N_HEADS(default: 8)]
	[--max_encoder_len MAX_ENCODER_LEN (default: 80)]
	[--max_decoder_len MAX_DECODER_LEN (default: 80)]	
	[--LAS_embed_dim LAS_EMBED_DIM (default: 512)]
	[--LAS_hidden_size LAS_HIDDEN_SIZE (default: 512)]
	[--num_epochs NUM_EPOCHS (default: 500)] 
	[--lr LR default=0.0005]    
	[--gradient_acc_steps GRADIENT_ACC_STEPS (default: 2)]
	[--max_norm MAX_NORM (default: 1.0)] 
	[--T_max T_MAX (default: 5000)] 
	[--model_no MODEL_NO (default: 0 (0: Transformer))]  
	[--train TRAIN (default:1)]  
	[--infer INFER (default: 0 (Infer input sentence labels from 	trained model))]


```

Or, if used as a package,
```python
from nlptoolkit.utils.config import Config
from nlptoolkit.punctuation_restoration.trainer import train_and_fit
from nlptoolkit.punctuation_restoration.infer import infer_from_trained

config = Config(task='punctuation_restoration') # loads default argument parameters as above
config.data_path = "./data/train.tags.en-fr.en"' # sets training data path
config.batch_size = 32
config.lr = 5e-5 # change learning rate
config.model_no = 1 # sets model to PuncLSTM
train_and_fit(config) # starts training with configured parameters
inferer = infer_from_trained(config) # initiate infer object, which loads the model for inference, after training model
inferer.infer_from_input() # infer from user console input
inferer.infer_from_file(in_file="./data/input.txt", out_file="./data/output.txt") # infer from input file
```

```python
inferer.infer_from_input()
```
Sample output:
```bash
Input sentence to punctuate:
hi how are you
Predicted Label:  Hi. How are you?

Input sentence to punctuate:
this is good thank you very much
Predicted Label:  This is good. Thank you very much.
```

### Pre-trained models
Download and zip contents of downloaded folder into ./data/ folder.
1. [PuncLSTM](https://drive.google.com/drive/folders/1ftDQYj3wv0t9MVtAVod5RIDMrY-NhZ82?usp=sharing) (includes preprocessed data, vocab, and saved results files)
---

## 7) Named Entity Recognition
In Named entity recognition (NER), the task is to recognise entities such as persons, organisations. Current models for this task: 
1. BERT

### Format of dataset files
Dataset format for both train & test follows the Conll2003 dataset format. Specifically, each row in the .txt file follows the following format:
```bash
EU NNP I-NP I-ORG
rejects VBZ I-VP O
German JJ I-NP I-MISC
call NN I-NP O
to TO I-VP O
boycott VB I-VP O
British JJ I-NP I-MISC
lamb NN I-NP O
. . O O
```
Here, the first column represents the word within the sentence, second column represents the parts-of-speech tag (not used), third column represents the tree chunk tag (not used), the fourth column is the NER tag. Only the first and fourth columns are used for this task and the rest are ignored. (A placeholder is still required for the second and third columns)

- Conll2003 dataset can be downloaded [here.](https://drive.google.com/drive/folders/1LAwi1TKTTfnG5ZPcbBPdyrreyxzAsjpi?usp=sharing)

### Running the model
Run ner.py

```bash
ner.py [-h] 
	[--train_path TRAIN_PATH] 
	[--test_path TEST_PATH]
	[--num_classes NUM_CLASSES]
	[--batch_size BATCH_SIZE]
	[--tokens_length TOKENS_LENGTH]
	[--max_steps MAX_STEPS]
	[--warmup_steps WARMUP_STEPS]
	[--weight_decay WEIGHT_DECAY]
	[--adam_epsilon ADAM_EPSILON]
	[--gradient_acc_steps GRADIENT_ACC_STEPS]
	[--num_epochs NUM_EPOCHS]
	[--lr LR]
	[--model_no MODEL_NO]
	[--model_type MODEL_TYPE]
	[--train TRAIN (default:1)]  
	[--evaluate EVALUATE (default:0)]
```

Or if used as a package:
```python
from nlptoolkit.utils.config import Config
from nlptoolkit.ner.trainer import train_and_fit
from nlptoolkit.ner.infer import infer_from_trained

config = Config(task='ner') # loads default argument parameters as above
config.train_path = './data/ner/conll2003/eng.train.txt' # sets training data path
config.test_path = './data/ner/conll2003/eng.testa.txt' # sets test data path
config.num_classes = 9 # sets number of NER classes
config.batch_size = 8
config.lr = 5e-5 # change learning rate
config.model_no = 0 # sets model to BERT
train_and_fit(config) # starts training with configured parameters
inferer = infer_from_trained(config) # initiate infer object, which loads the model for inference, after training model
inferer.infer_from_input() # infer from user console input
inferer.infer_from_file(in_file="./data/input.txt", out_file="./data/output.txt")
```

```python
inferer.infer_from_input()
```
Sample output:
```bash
Type input sentence: ('quit' or 'exit' to terminate)
John took a flight from Singapore to China, but stopped by Japan along the way.
Words --- Tags:
john (I-PER) 
took (O) 
a (O) 
flight (O) 
from (O) 
singapore (I-LOC) 
to (O) 
china, (I-LOC) 
but (O) 
stopped (O) 
by (O) 
japan (I-LOC) 
along (O) 
the (O) 
way. (O) 
```

### Pre-trained models
Download and zip contents of downloaded folder into ./data/ folder.
1. [BERT](https://drive.google.com/drive/folders/1-srmwPo23MGfmc7x80Ojb4_4sPCFeGNo?usp=sharing) (includes preprocessed data, vocab, and saved results files)
---

# Benchmark Results

## 1) Classification (IMDB dataset : 25000 train, 25000 test data points)

### Fine-tuned XLNet English Model (12-layer, 768-hidden, 12-heads, 110M parameters)  
![](https://github.com/plkmo/NLP_Toolkit/blob/master/results/imdb/classification/loss_vs_epoch_2.png) 

![](https://github.com/plkmo/NLP_Toolkit/blob/master/results/imdb/classification/accuracy_vs_epoch_2.png) 

### Fine-tuned BERT English Model (12-layer, 768-hidden, 12-heads, 110M parameters)  
![](https://github.com/plkmo/NLP_Toolkit/blob/master/results/imdb/classification/loss_vs_epoch_1.png) 

![](https://github.com/plkmo/NLP_Toolkit/blob/master/results/imdb/classification/accuracy_vs_epoch_1.png) 

## 6) Punctuation Restoration (TED dataset)

### Punc-LSTM (Embedding dim=512, LSTM hidden size=512)
![](https://github.com/plkmo/NLP_Toolkit/blob/master/results/TED_punctuation/test_loss_vs_epoch_1.png) 

![](https://github.com/plkmo/NLP_Toolkit/blob/master/results/TED_punctuation/test_Accuracy_vs_epoch_1.png) 

## 7) Named Entity Recognition (Conll2003 dataset)

### Fine-tuned BERT English Model (uncased, 12-layer, 768-hidden, 12-heads, 110M parameters)  

![](https://github.com/plkmo/NLP_Toolkit/blob/master/results/conll2003/ner/test_loss_vs_epoch_0.png) 

![](https://github.com/plkmo/NLP_Toolkit/blob/master/results/conll2003/ner/test_Accuracy_vs_epoch_0.png) 

---

# References
1. Attention Is All You Need, Vaswani et al, https://arxiv.org/abs/1706.03762
2. Graph Convolutional Networks for Text Classification, Liang Yao et al, https://arxiv.org/abs/1809.05679
3. Speech-Transformer: A No-Recurrence Sequence-To-Sequence Model For Speech Recognition, Linhao Dong et al, https://ieeexplore.ieee.org/document/8462506
4. Listen, Attend and Spell, William Chan et al, https://arxiv.org/abs/1508.01211
5. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Devlin et al, https://arxiv.org/abs/1810.04805
6. XLNet: Generalized Autoregressive Pretraining for Language Understanding, Yang et al, https://arxiv.org/abs/1906.08237
7. Investigating LSTM for punctuation prediction, Xu et al, https://ieeexplore.ieee.org/document/7918492
8. HuggingFace's Transformers: State-of-the-art Natural Language Processing, Thomas Wolf et al, https://arxiv.org/abs/1910.03771

---

# To do list
In order of priority:
- [ ] Include package usage info for ~~classification~~, ASR, summarization, ~~translation~~, ~~generation~~, ~~punctuation_restoration~~, ~~NER~~, POS
- [ ] Include benchmark results for  ~~classification~~, ASR, summarization, translation, generation, ~~punctuation_restoration~~, ~~NER~~, POS
- [ ] Include pre-trained models + demo based on benchmark datasets for ~~classification~~, ASR, summarization, translation, ~~generation~~, punctuation_restoration, NER, POS
- [ ] Include more models for punctuation restoration, translation, NER

