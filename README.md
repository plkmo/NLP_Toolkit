# NLP Toolkit
Library containing state-of-the-art models for Natural Language Processing tasks

## Tasks
1. Classification
2. Automatic Speech Recognition
3. Text Summarization
4. Machine Translation

## 1) Classification
The goal of classification is to segregate documents into appropriate classes based on their text content. Currently, the classification toolkit uses text-based Graph Convolution Networks (GCN) and Bidirectional Encoder Representations from Transformers (BERT).

### Format of datasets files
The training data (default: train.csv) should be formatted into two columns “text” and “label” respectively, with rows being the documents index. “text” contains the raw text and “label” contains the corresponding label (integers 0, 1, 2… depending on the number of classes)

The infer data (default: infer.csv) should be formatted into at least one column “text” being the raw text and rows being the documents index. Optional column “label” can be added and --train_test_split argument set to 1 to use infer.csv as the test set for model verification.

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
            [--model_no MODEL_NO]
```
The script outputs a results.csv file containing the indexes of the documents in infer.csv and their corresponding predicted labels.

## 2) Automatic Speech Recognition
Automatic Speech Recognition (ASR) aims to convert audio signals into text. This library contains the following models for ASR: Speech-Transformer, Listen-Attend-Spell (LAS).

### Format of dataset files
The folder containing the dataset should have the following structure: folder/speaker/chapter
Within the chapter subdirectory, the audio files (in .flac format) are named speaker-chapter-file_id (file_id In running order)
The transcript .txt file for the files within the chapter should be located in the chapter subdirectory. In the transcript file, each row should consist of the speaker-chapter-file_id (space) transcript.

### Running the model
Run speech.py with arguments below

## 3) Summarization
Text summarization aims to distill a paragraph chunk into a few sentences that capture the essential information. This library contains the following models for text summarization: Convolutional Transformer, Seq2Seq (LAS architecture)

### Format of dataset files

## 4) Machine Translation
The goal of machine translation is to translate text from one form of language to another. This library contains the Transformer model to accomplish this.

### Format of dataset files
