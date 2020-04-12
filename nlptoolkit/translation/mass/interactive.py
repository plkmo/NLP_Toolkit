#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""
import os
import jieba
from collections import namedtuple
from .xmasked_seq2seq import XMassTranslationTask
from .xtransformer import XTransformerModel
import fileinput
import re
from tqdm import tqdm

import torch

from fairseq import checkpoint_utils, options, tasks, utils


Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, task, max_positions, encode_fn):
    tokens = [
        task.source_dictionary.encode_line(
            encode_fn(src_str), add_if_not_exist=False
        ).long()
        for src_str in lines
    ]
    lengths = torch.LongTensor([t.numel() for t in tokens])
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch['id'],
            src_tokens=batch['net_input']['src_tokens'], src_lengths=batch['net_input']['src_lengths'],
        )


def main(args):
    utils.import_user_module(args)

    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Initialize generator
    generator = task.build_generator(args)

    # Hack to support GPT-2 BPE
    if args.remove_bpe == 'gpt2':
        from fairseq.gpt2_bpe.gpt2_encoding import get_encoder
        decoder = get_encoder(
            'fairseq/gpt2_bpe/encoder.json',
            'fairseq/gpt2_bpe/vocab.bpe',
        )
        encode_fn = lambda x: ' '.join(map(str, decoder.encode(x)))
    else:
        decoder = None
        encode_fn = lambda x: x

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    )

    if args.buffer_size > 1:
        print('| Sentence buffer size:', args.buffer_size)
    print('| Type the input sentence and press return:')
    start_id = 0
    for inputs in buffered_read(args.input, args.buffer_size):
        results = []
        for batch in make_batches(inputs, args, task, max_positions, encode_fn):
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            if use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()

            sample = {
                'net_input': {
                    'src_tokens': src_tokens,
                    'src_lengths': src_lengths,
                },
            }
            translations = task.inference_step(generator, models, sample)
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                results.append((start_id + id, src_tokens_i, hypos))

        # sort output to match input order
        for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
            if src_dict is not None:
                src_str = src_dict.string(src_tokens, args.remove_bpe)
                print('S-{}\t{}'.format(id, src_str))

            # Process top predictions
            for hypo in hypos[:min(len(hypos), args.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe,
                )
                if decoder is not None:
                    hypo_str = decoder.decode(map(int, hypo_str.strip().split()))
                print('H-{}\t{}\t{}'.format(id, hypo['score'], hypo_str))
                print('P-{}\t{}'.format(
                    id,
                    ' '.join(map(lambda x: '{:.4f}'.format(x), hypo['positional_scores'].tolist()))
                ))
                if args.print_alignment:
                    print('A-{}\t{}'.format(
                        id,
                        ' '.join(map(lambda x: str(utils.item(x)), alignment))
                    ))

        # update running id counter
        start_id += len(inputs)


def cli_main():
    parser = options.get_generation_parser(interactive=True)
    args = options.parse_args_and_arch(parser)
    main(args)

class Translator(object):
    def __init__(self, data_path="./data/processed", \
                 checkpoint_path="./checkpoints/zhen_mass_pre-training.pt",\
                 task='xmasked_seq2seq',\
                 user_dir='mass',\
                 s='zh', t='en',\
                 langs='en,zh',\
                 mt_steps='zh-en',\
                 source_langs='zh',\
                 target_langs='en',\
                 beam=5,\
                 use_cuda=1):
        self.parser = options.get_generation_parser(interactive=True)
        self.parser.set_defaults(path=checkpoint_path, task=task, user_dir=user_dir, s=s, t=t,\
                                 source_langs=source_langs, target_langs=target_langs,\
                                 langs=langs, mt_steps=mt_steps, beam=beam)
        self.use_cuda = use_cuda
        self.args = options.parse_args_and_arch(self.parser,\
                                               input_args=[data_path])
        self.args.user_dir = user_dir
        self.args.s = s
        self.args.t = t
        self.args.langs = langs
        self.args.mt_steps = mt_steps
        self.args.source_langs = source_langs
        self.args.target_langs = target_langs
        self.args.remove_bpe = '@@ '
        #self.args, _ = self.parser.parse_known_args([data_path])
        
        utils.import_user_module(self.args)

        if self.args.buffer_size < 1:
            self.args.buffer_size = 1
        if self.args.max_tokens is None and self.args.max_sentences is None:
            self.args.max_sentences = 1
    
        assert not self.args.sampling or self.args.nbest == self.args.beam, \
            '--sampling requires --nbest to be equal to --beam'
        assert not self.args.max_sentences or self.args.max_sentences <= self.args.buffer_size, \
            '--max-sentences/--batch-size cannot be larger than --buffer-size'
    
        print(self.args)
    
        #self.use_cuda = torch.cuda.is_available() and not self.args.cpu
    
        # Setup task, e.g., translation
        self.task = tasks.setup_task(self.args)
        
        # Load ensemble
        print('| loading model(s) from {}'.format(self.args.path))
        self.models, self._model_args = checkpoint_utils.load_model_ensemble(
            self.args.path.split(':'),
            arg_overrides=eval(self.args.model_overrides),
            task=self.task,
        )
    
        # Set dictionaries
        self.src_dict = self.task.source_dictionary
        self.tgt_dict = self.task.target_dictionary
    
        # Optimize ensemble for generation
        for model in self.models:
            model.make_generation_fast_(
                beamable_mm_beam_size=None if self.args.no_beamable_mm else self.args.beam,
                need_attn=self.args.print_alignment,
            )
            if self.args.fp16:
                model.half()
            if self.use_cuda:
                model.cuda()
    
        # Initialize generator
        self.generator = self.task.build_generator(self.args)
    
        # Hack to support GPT-2 BPE
        if self.args.remove_bpe == 'gpt2':
            from fairseq.gpt2_bpe.gpt2_encoding import get_encoder
            self.decoder = get_encoder(
                'fairseq/gpt2_bpe/encoder.json',
                'fairseq/gpt2_bpe/vocab.bpe',
            )
            self.encode_fn = lambda x: ' '.join(map(str, self.decoder.encode(x)))
        else:
            self.decoder = None
            self.encode_fn = lambda x: x
    
        # Load alignment dictionary for unknown word replacement
        # (None if no unknown word replacement, empty if no path to align dictionary)
        self.align_dict = utils.load_align_dict(self.args.replace_unk)
    
        self.max_positions = utils.resolve_max_positions(
            self.task.max_positions(),
            *[model.max_positions() for model in self.models]
        )
    
        if self.args.buffer_size > 1:
            print('| Sentence buffer size:', self.args.buffer_size)
    
    def translate(self, sent, verbose=False):
        start_id = 0
        if self.args.s == 'zh':
            sent = re.sub(' +', '', sent)
            sent = jieba.tokenize(sent)
            sent = " ".join(s[0] for s in sent)
        inputs = [sent]
        results = []
        for batch in make_batches(inputs, self.args, self.task, self.max_positions, self.encode_fn):
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            if self.use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()

            sample = {
                'net_input': {
                    'src_tokens': src_tokens,
                    'src_lengths': src_lengths,
                },
            }
            translations = self.task.inference_step(self.generator, self.models, sample)
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], self.tgt_dict.pad())
                results.append((start_id + id, src_tokens_i, hypos))

        # sort output to match input order
        for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
            if self.src_dict is not None:
                src_str = self.src_dict.string(src_tokens, self.args.remove_bpe)
                if verbose:
                    print('S-{}\t{}'.format(id, src_str))

            # Process top predictions
            for hypo in hypos[:min(len(hypos), self.args.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                    align_dict=self.align_dict,
                    tgt_dict=self.tgt_dict,
                    remove_bpe=self.args.remove_bpe,
                )
                if self.decoder is not None:
                    hypo_str = self.decoder.decode(map(int, hypo_str.strip().split()))
                hypo_str = self.corrector_module(hypo_str)
                if verbose:
                    print('H-{}\t{}\t{}'.format(id, hypo['score'], hypo_str))
                    print('P-{}\t{}'.format(
                        id,
                        ' '.join(map(lambda x: '{:.4f}'.format(x), hypo['positional_scores'].tolist()))
                    ))
                    if self.args.print_alignment:
                        print('A-{}\t{}'.format(
                            id,
                            ' '.join(map(lambda x: str(utils.item(x)), alignment))
                        ))
        return hypo_str
    
    @staticmethod
    def repeats(string):
        for x in range(1, len(string)):
            substring = string[:x]
    
            if substring * (len(string)//len(substring))+(substring[:len(string)%len(substring)]) == string:
                return substring
        return string
    
    def corrector_module(self, sent):
        sent = self.repeats(sent) # remove repeated strings
        sent = re.sub(r'@(.+)@', r'\1', sent) # remove @ annotations
        sent = re.sub(" +([!\.\?,])", r"\1", sent) # remove extra spaces
        sent = re.sub("([!\?\.,()])+[!\?\.,()]+", '', sent) # remove consecutive symbols
        sent = re.sub(" &apos;", "'", sent) # convert apostrophe
        sent = re.sub(" +$", '', sent) # remove extra spaces from end of string
        return sent
        
    def translate_file(self, in_file="./data/input.txt", out_file="./data/output.txt", \
                       compare=False,\
                       labels_path=''):
        
        with open(in_file, "r", encoding='utf8') as f:
            sents = f.readlines()
        
        if os.path.isfile(labels_path):
            with open(labels_path, 'r', encoding='utf8') as f:
                labels = f.readlines()
            assert len(sents) == len(labels)
        
        with open(out_file, "w", encoding='utf8') as f:
            if compare:
                f.write('Sentence -- Translated\n')
            
            for idx, sent in tqdm(enumerate(sents), total=len(sents)):
                translated_sent = self.translate(sent)
            
                if compare:
                    if os.path.isfile(labels_path):
                        f.write(sent + ' -- ' + translated_sent +'\n')
                        f.write('Ground truth: ' + labels[idx] +'\n')
                    else:
                        f.write(sent + ' -- ' + translated_sent +'\n')
                else:
                    f.write(translated_sent + '\n')
                    if os.path.isfile(labels_path):
                        f.write('Ground truth: ' + labels[idx] +'\n')
        print('Done and saved!')

    def translate_from_input(self):
        while True:
            sent = input("Type input sentence: (\'exit\' or \'quit\' to terminate)\n")
            if sent in ['quit', 'exit']:
                break
            translated_sent = self.translate(sent)
        return translated_sent
    
if __name__ == '__main__':
    translator = Translator()
    translator.translate("我们只能希望")
