# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:01:47 2019

@author: WT
"""
import torch
from torch.autograd import Variable
from nltk.translate import bleu_score
from .models.Transformer.Transformer import create_masks
from .train_funcs import load_model_and_optimizer
from .preprocessing_funcs import tokener, load_dataloaders
import time
    
def dum_tokenizer(sent):
    return sent.split()

def calculate_bleu(src, trg, weights=(0.25, 0.25, 0.25, 0.25), corpus_level=False):
    # src = [[sent words1], [sent words2], ...], trg = [sent words]
    if not corpus_level:
        score = bleu_score.sentence_bleu(src, trg, weights=weights)
    else:
        score = bleu_score.corpus_bleu(src, trg, weights=weights)
    return score

def evaluate_bleu(args):
    args.batch_size = 1
    #tokenizer_en = tokener("en")
    train_iter, FR, EN, train_length = load_dataloaders(args)
    src_vocab = len(EN.vocab)
    trg_vocab = len(FR.vocab)
    
    cuda = torch.cuda.is_available()
    net, _, _, _, _, _ = load_model_and_optimizer(args, src_vocab, \
                                                  trg_vocab, cuda)
    
    net.eval()
    trg_init = FR.vocab.stoi["<sos>"]
    trg_init = Variable(torch.LongTensor([trg_init])).unsqueeze(0)
    
    refs = []; hyps = []
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
            refs.append([stepwise_translated_words]) # need to remove <eos> tokens
            hyps.append([FR.vocab.itos[i] for i in labels])
    
    return

def infer(args, from_data=False):
    args.batch_size = 1
    tokenizer_en = tokener("en")
    train_iter, FR, EN, train_length = load_dataloaders(args)
    src_vocab = len(EN.vocab)
    trg_vocab = len(FR.vocab)
    
    cuda = torch.cuda.is_available()
    net, _, _, _, _, _ = load_model_and_optimizer(args, src_vocab, \
                                                  trg_vocab, cuda)
    '''
    ### Load model and vocab
    FR = torchtext.data.Field(tokenize=dum_tokenizer, lower=True, init_token="<sos>", eos_token="<eos>",\
                              batch_first=True)
    EN = torchtext.data.Field(tokenize=dum_tokenizer, lower=True, batch_first=True)
    train = torchtext.data.TabularDataset(os.path.join("./data/", "df.csv"), format="csv", \
                                             fields=[("EN", EN), ("FR", FR)])
    FR.build_vocab(train)
    EN.build_vocab(train)
    src_vocab = len(EN.vocab)
    trg_vocab = len(FR.vocab)
    
    cuda = torch.cuda.is_available()
    net = Transformer(src_vocab=src_vocab, trg_vocab=trg_vocab, d_model=512, num=6, n_heads=8)
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
    scheduler = CosineWithRestarts(optimizer, T_max=10)
    if cuda:
        net.cuda()
    start_epoch, acc = load_state(net, optimizer, scheduler, model_no=0, load_best=False)
    '''
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
                print("\n\nInput:")
                print(" ".join(EN.vocab.itos[i] for i in data.EN[0]))
                print("\nStepwise-translated:")
                print(" ".join(stepwise_translated_words))
                print("\nFinal step translated words: ")
                print(" ".join(final_step_words))
                print("\nGround Truth:")
                print(" ".join(FR.vocab.itos[i] for i in labels))
                print("Bleu score (stepwise-translated sentence level): %.3f" % score)
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