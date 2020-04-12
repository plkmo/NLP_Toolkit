# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:01:47 2019

@author: WT
"""
import pickle
import os
import torch
import pandas as pd
from torch.autograd import Variable
from nltk.translate import bleu_score
from torchnlp.metrics import get_moses_multi_bleu
from .models.Transformer.Transformer import create_masks
from .train_funcs import load_model_and_optimizer
from .preprocessing_funcs import tokener, load_dataloaders
from tqdm import tqdm
import time
import logging

tqdm.pandas(desc="prog_bar")
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def load_pickle(filename):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data
    
def dum_tokenizer(sent):
    return sent.split()

def calculate_bleu(src, trg, corpus_level=False, weights=(0.25, 0.25, 0.25, 0.25), use_torchnlp=True):
    # src = [[sent words1], [sent words2], ...], trg = [sent words]
    if not use_torchnlp:
        if not corpus_level:
            score = bleu_score.sentence_bleu(src, trg, weights=weights)
        else:
            score = bleu_score.corpus_bleu(src, trg, weights=weights)
    else:
        score = get_moses_multi_bleu(src, trg, lowercase=True)
    return score

def evaluate_corpus_bleu(args, early_stopping=True, stop_no=1000):
    args.batch_size = 1
    train_iter, FR, EN, train_length = load_dataloaders(args)
    src_vocab = len(EN.vocab)
    trg_vocab = len(FR.vocab)
    
    cuda = torch.cuda.is_available()
    if args.fp16:    
        from apex import amp
    else:
        amp = None
    net, _, _, _, _, _ = load_model_and_optimizer(args, src_vocab, \
                                                  trg_vocab, cuda, amp=amp)
    
    net.eval()
    trg_init = FR.vocab.stoi["<sos>"]
    trg_init = Variable(torch.LongTensor([trg_init])).unsqueeze(0)
    
    logger.info("Evaluating corpus bleu...")
    refs = []; hyps = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(train_iter), total=len(train_iter)):
            trg_input = trg_init
            labels = data.FR[:,1:].contiguous().view(-1)
            src_mask, trg_mask = create_masks(data.EN, trg_input)
            if cuda:
                data.EN = data.EN.cuda(); trg_input = trg_input.cuda(); labels = labels.cuda()
                src_mask = src_mask.cuda(); trg_mask = trg_mask.cuda()
            stepwise_translated_words, final_step_words = net(data.EN, trg_input, src_mask, None,\
                                                              infer=True, trg_vocab_obj=FR)
            refs.append([stepwise_translated_words]) # need to remove <eos> tokens
            hyps.append([FR.vocab.itos[i] for i in labels[:-1]])
            if early_stopping and ((i + 1) % stop_no == 0):
                print(refs); print(hyps)
                break
    score = calculate_bleu(refs, hyps, corpus_level=True)
    print("Corpus bleu score: %.5f" % score)
    return score

class infer_from_trained(object):
    def __init__(self, args=None):
        if args is None:
            self.args = load_pickle("args.pkl")
        else:
            self.args = args
        self.cuda = torch.cuda.is_available()
        self.args.batch_size = 1
        
        if self.args.model_no != 1:
            logger.info("Loading tokenizer and model...")
            self.tokenizer_en = tokener(args.src_lang)
            train_iter, FR, EN, train_length = load_dataloaders(self.args)
            self.FR = FR
            self.EN = EN
            self.train_iter = train_iter
            self.train_length = train_length
            self.src_vocab = len(EN.vocab)
            self.trg_vocab = len(FR.vocab)
            
            if self.args.fp16:    
                from apex import amp
            else:
                amp = None
            self.amp = amp
            net, _, _, _, _, _ = load_model_and_optimizer(self.args, self.src_vocab, \
                                                          self.trg_vocab, self.cuda, amp=amp)
            self.net = net
            self.net.eval()
            trg_init = FR.vocab.stoi["<sos>"]
            self.trg_init = Variable(torch.LongTensor([trg_init])).unsqueeze(0)
        elif self.args.model_no == 1:
            from .mass.interactive import Translator
            src, tgt = "zh-en".split('-')
            logger.info("Loading translator, tokenizer...")
            self.translator = Translator(data_path='./data/data-bin/processed_data_%s_%s' % (src, tgt),\
                                         checkpoint_path="./data/checkpoints/%s_%s/checkpoint50.pt" % (src, tgt),\
                                         task='translation',\
                                         user_dir='',\
                                         s=src, t=tgt,\
                                         langs='%s,%s' % (src, tgt),\
                                         mt_steps='%s-%s' % (src, tgt),\
                                         source_langs=src,\
                                         target_langs=tgt,\
                                         beam=5,\
                                         use_cuda=args.cuda)
    
    def infer_sentence(self, sent):
        if self.args.model_no != 1:
            sent = self.tokenizer_en.tokenize(sent).split()
            sent = [self.EN.vocab.stoi[tok] for tok in sent]
            sent = Variable(torch.LongTensor(sent)).unsqueeze(0)
            
            trg = self.trg_init
            src_mask, _ = create_masks(sent, self.trg_init)
            if self.cuda:
                sent = sent.cuda(); src_mask = src_mask.cuda()
                trg = trg.cuda()
                
            with torch.no_grad():
                stepwise_translated_words, final_step_words = self.net(sent, trg, src_mask, None, \
                                                                  infer=True, trg_vocab_obj=self.FR)
                
            stepwise_translated = " ".join(stepwise_translated_words)
            final_translated = " ".join(final_step_words)
            print("Stepwise-translated:")
            print(stepwise_translated)
            print("\nFinal step translated words: ")
            print(final_translated)
            return stepwise_translated, final_translated
        elif self.args.model_no == 2:
            translated = self.translator.translate(sent)
            return translated
        
    def infer_from_input(self):
        self.net.eval()
        while True:
            user_input = input("Type input sentence (Type \'exit' or \'quit' to quit):\n")
            if user_input in ["exit", "quit"]:
                break
            predicted = self.infer_sentence(user_input)
        return predicted
    
    def infer_from_file(self, in_file="./data/input.txt", out_file="./data/output.txt"):
        df = pd.read_csv(in_file, header=None, names=["sents"])
        df['labels'] = df.progress_apply(lambda x: self.infer_sentence(x['sents']), axis=1)
        df.to_csv(out_file, index=False)
        logger.info("Done and saved as %s!" % out_file)
        return

def infer(args, from_data=False):
    args.batch_size = 1
    tokenizer_en = tokener("en")
    train_iter, FR, EN, train_length = load_dataloaders(args)
    src_vocab = len(EN.vocab)
    trg_vocab = len(FR.vocab)
    
    cuda = torch.cuda.is_available()
    if args.fp16:    
        from apex import amp
    else:
        amp = None
    net, _, _, _, _, _ = load_model_and_optimizer(args, src_vocab, \
                                                  trg_vocab, cuda, amp=amp)
    
    net.eval()
    trg_init = FR.vocab.stoi["<sos>"]
    trg_init = Variable(torch.LongTensor([trg_init])).unsqueeze(0)
    
    if from_data:
        with torch.no_grad():
            for i, data in enumerate(train_iter):
                trg_input = trg_init
                labels = data.FR[:,1:].contiguous().view(-1)
                src_mask, trg_mask = create_masks(data.EN, trg_input)
                if cuda:
                    data.EN = data.EN.cuda(); trg_input = trg_input.cuda(); labels = labels.cuda()
                    src_mask = src_mask.cuda(); trg_mask = trg_mask.cuda()
                stepwise_translated_words, final_step_words = net(data.EN, trg_input, src_mask, None,\
                                                                  infer=True, trg_vocab_obj=FR)
                score = calculate_bleu([stepwise_translated_words], [FR.vocab.itos[i] for i in labels])
                print([stepwise_translated_words]); print([FR.vocab.itos[i] for i in labels])
                print("\n\nInput:")
                print(" ".join(EN.vocab.itos[i] for i in data.EN[0]))
                print("\nStepwise-translated:")
                print(" ".join(stepwise_translated_words))
                print("\nFinal step translated words: ")
                print(" ".join(final_step_words))
                print("\nGround Truth:")
                print(" ".join(FR.vocab.itos[i] for i in labels))
                print("Bleu score (stepwise-translated sentence level): %.5f" % score)
                time.sleep(7)
    
    else:
    
        while True:
        ### process user input sentence
            sent = input("Enter English sentence:\n")
            if sent == "quit":
                break
            sent = tokenizer_en.tokenize(sent).split()
            sent = [EN.vocab.stoi[tok] for tok in sent]
            sent = Variable(torch.LongTensor(sent)).unsqueeze(0)
            
            trg = trg_init
            src_mask, _ = create_masks(sent, trg_init)
            if cuda:
                sent = sent.cuda(); src_mask = src_mask.cuda()
                trg = trg.cuda()
                
            with torch.no_grad():
                stepwise_translated_words, final_step_words = net(sent, trg, src_mask, None, \
                                                                  infer=True, trg_vocab_obj=FR)
                
                '''
                e_out = net.encoder(sent, src_mask) # encoder output for english sentence
                translated_word = []; translated_word_idxs = []
                for i in range(2, 80):
                    trg_mask = create_trg_mask(trg, cuda=cuda)
                    if cuda:
                        trg = trg.cuda(); trg_mask = trg_mask.cuda()
                    outputs = net.fc1(net.decoder(trg, e_out, src_mask, trg_mask))
                    out_idxs = torch.softmax(outputs, dim=2).max(2)[1]
                    trg = torch.cat((trg, out_idxs[:,-1:]), dim=1)
                    if cuda:
                        out_idxs = out_idxs.cpu().numpy()
                    else:
                        out_idxs = out_idxs.numpy()
                    translated_word_idxs.append(out_idxs.tolist()[0][-1])
                    if translated_word_idxs[-1] == FR.vocab.stoi["<eos>"]:
                        break
                    translated_word.append(FR.vocab.itos[translated_word_idxs[-1]])
                
            print(" ".join(translated_word))
            print(" ".join(FR.vocab.itos[i] for i in out_idxs[0][:-1]))
            '''
            print("Stepwise-translated:")
            print(" ".join(stepwise_translated_words))
            print("\nFinal step translated words: ")
            print(" ".join(final_step_words))