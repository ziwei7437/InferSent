# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import sys
import time
import argparse

import logging

import numpy as np

import torch
#from torch.autograd import Variable
from torch import optim
import torch.nn as nn

#from data import get_nli, get_batch, build_vocab
from tasks import get_task, MnliMismatchedProcessor
#from mutils import get_optimizer
from models import InferSent, SimpleClassifier

import initialization
from runner import RunnerParameters, GlueTaskClassifierRunner


def get_args(*in_args):
    parser = argparse.ArgumentParser(description='NLI training')
    # === Required Parameters ===
    # paths
    parser.add_argument("--data_dir",
                        type=str,
                        default=None,
                        required=True,
                        help="training dataset directory")
    parser.add_argument("--task_name",
                        type=str,
                        default=None,
                        required=True,
                        help='the name of the task to train.')
    parser.add_argument("--output_dir", 
                        type=str, 
                        default=None,
                        required=True,
                        help="Output directory")
    parser.add_argument("--word_emb_path",
                        type=str, 
                        required=True,
                        default="dataset/GloVe/glove.840B.300d.txt", 
                        help="word embedding file path")
    parser.add_argument("--model_path",
                        type=str,
                        required=True,
                        default="encoder/infersent1.pkl",
                        help="state dict of pre-trained infersent models")

    # === Optional Parameters ===
    # training
    # we dont train the encoder.
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
    parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
    parser.add_argument("--nonlinear_fc", type=float, default=0, help="use nonlinearity in fc")
    parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1", help="adam or sgd,lr=0.1")
    parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
    parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
    parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
    parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")

    # tasks
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_val", action="store_true")

    # training args for classifier
    parser.add_argument("--force-overwrite", action="store_true")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")

    # model
    parser.add_argument("--enc_lstm_dim", type=int, default=2048, help="encoder nhid dimension")
    parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
    parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
    parser.add_argument("--n_classes", type=int, default=3, help="entailment/neutral/contradiction")
    parser.add_argument("--pool_type", type=str, default='max', help="max or mean")
    parser.add_argument("--dropout_prob", type=float, default=0.1, help="classifier hidden dropout probability")
    parser.add_argument("--model_version", type=int, default=1, help="model version to use")
    parser.add_argument("--k_freq_words", type=int, default=100000, help="k most frequent words")


    # gpu
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--seed", type=int, default=-1, help="seed")
    parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")

    # data
    parser.add_argument("--word_emb_dim", type=int, default=300, help="word embedding dimension")
    
    args = parser.parse_args(*in_args)
    return args


def print_args(args):
    for k, v in vars(args).items():
        print("  {}: {}".format(k, v))


def trainepoch(epoch):
    print('\nTRAINING : Epoch ' + str(epoch))
    nli_net.train()
    all_costs = []
    logs = []
    words_count = 0

    last_time = time.time()
    correct = 0.
    # shuffle the data
    permutation = np.random.permutation(len(train['s1']))

    s1 = train['s1'][permutation]
    s2 = train['s2'][permutation]
    target = train['label'][permutation]


    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * params.decay if epoch>1\
        and 'sgd' in params.optimizer else optimizer.param_groups[0]['lr']
    print('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))

    for stidx in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[stidx:stidx + params.batch_size],
                                     word_vec, params.word_emb_dim)
        s2_batch, s2_len = get_batch(s2[stidx:stidx + params.batch_size],
                                     word_vec, params.word_emb_dim)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        tgt_batch = Variable(torch.LongTensor(target[stidx:stidx + params.batch_size])).cuda()
        k = s1_batch.size(1)  # actual batch size

        # model forward
        output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()
        assert len(pred) == len(s1[stidx:stidx + params.batch_size])

        # loss
        loss = loss_fn(output, tgt_batch)
        all_costs.append(loss.data[0])
        words_count += (s1_batch.nelement() + s2_batch.nelement()) / params.word_emb_dim

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping (off by default)
        shrink_factor = 1
        total_norm = 0

        for p in nli_net.parameters():
            if p.requires_grad:
                p.grad.data.div_(k)  # divide by the actual batch size
                total_norm += p.grad.data.norm() ** 2
        total_norm = np.sqrt(total_norm)

        if total_norm > params.max_norm:
            shrink_factor = params.max_norm / total_norm
        current_lr = optimizer.param_groups[0]['lr'] # current lr (no external "lr", for adam)
        optimizer.param_groups[0]['lr'] = current_lr * shrink_factor # just for update

        # optimizer step
        optimizer.step()
        optimizer.param_groups[0]['lr'] = current_lr

        if len(all_costs) == 100:
            logs.append('{0} ; loss {1} ; sentence/s {2} ; words/s {3} ; accuracy train : {4}'.format(
                            stidx, round(np.mean(all_costs), 2),
                            int(len(all_costs) * params.batch_size / (time.time() - last_time)),
                            int(words_count * 1.0 / (time.time() - last_time)),
                            round(100.*correct/(stidx+k), 2)))
            print(logs[-1])
            last_time = time.time()
            words_count = 0
            all_costs = []
    train_acc = round(100 * correct/len(s1), 2)
    print('results : epoch {0} ; mean accuracy train : {1}'
          .format(epoch, train_acc))
    return train_acc


def evaluate(epoch, eval_type='valid', final_eval=False):
    nli_net.eval()
    correct = 0.
    global val_acc_best, lr, stop_training, adam_stop

    if eval_type == 'valid':
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    s1 = valid['s1'] if eval_type == 'valid' else test['s1']
    s2 = valid['s2'] if eval_type == 'valid' else test['s2']
    target = valid['label'] if eval_type == 'valid' else test['label']

    for i in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[i:i + params.batch_size], word_vec, params.word_emb_dim)
        s2_batch, s2_len = get_batch(s2[i:i + params.batch_size], word_vec, params.word_emb_dim)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        tgt_batch = Variable(torch.LongTensor(target[i:i + params.batch_size])).cuda()

        # model forward
        output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()

    # save model
    eval_acc = round(100 * correct / len(s1), 2)
    if final_eval:
        print('finalgrep : accuracy {0} : {1}'.format(eval_type, eval_acc))
    else:
        print('togrep : results : epoch {0} ; mean accuracy {1} :\
              {2}'.format(epoch, eval_type, eval_acc))

    if eval_type == 'valid' and epoch <= params.n_epochs:
        if eval_acc > val_acc_best:
            print('saving model at epoch {0}'.format(epoch))
            if not os.path.exists(params.outputdir):
                os.makedirs(params.outputdir)
            torch.save(nli_net.state_dict(), os.path.join(params.outputdir,
                       params.outputmodelname))
            val_acc_best = eval_acc
        else:
            if 'sgd' in params.optimizer:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / params.lrshrink
                print('Shrinking lr by : {0}. New lr = {1}'
                      .format(params.lrshrink,
                              optimizer.param_groups[0]['lr']))
                if optimizer.param_groups[0]['lr'] < params.minlr:
                    stop_training = True
            if 'adam' in params.optimizer:
                # early stopping (at 2nd decrease in accuracy)
                stop_training = adam_stop
                adam_stop = True
    return eval_acc


def main():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    args = get_args()
    print_args(args)
    device, n_gpu = initialization.init_cuda_from_args(args, logger=logger)
    initialization.init_seed(args, n_gpu=n_gpu, logger=logger)
    initialization.init_train_batch_size(args)
    initialization.init_output_dir(args)
    initialization.save_args(args)
    task = get_task(args.task_name, args.data_dir)


    # model config
    config = {
        'word_emb_dim'   :  args.word_emb_dim   ,
        'enc_lstm_dim'   :  args.enc_lstm_dim   ,
        'n_enc_layers'   :  args.n_enc_layers   ,
        'dpout_model'    :  args.dpout_model    ,
        'dpout_fc'       :  args.dpout_fc       ,
        'fc_dim'         :  args.fc_dim         ,
        'bsize'          :  args.batch_size     ,
        'n_classes'      :  args.n_classes      ,
        'pool_type'      :  args.pool_type      ,
        'nonlinear_fc'   :  args.nonlinear_fc   ,
        'use_cuda'       :  not args.no_cuda    ,
        'version'        :  args.model_version  ,
        'dropout_prob'   :  args.dropout_prob   ,
    }

    # load model
    model = InferSent(config)
    model.load_state_dict(torch.load(args.model_path))
    model = model.cuda() if not args.no_cuda else model
    model.set_w2v_path(args.word_emb_path)
    model.build_vocab_k_words(k=args.k_freq_words)

    # load classifier
    classifier = SimpleClassifier(config)

    # get train examples
    train_examples = task.get_train_examples()
    # calculate t_total
    t_total = initialization.get_opt_train_steps(len(train_examples), args)


    # build optimizer.
    optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)

    # create running parameters
    r_params = RunnerParameters(
        local_rank=args.local_rank,
        n_gpu=n_gpu,
        learning_rate=5e-5,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        t_total=t_total,
        warmup_proportion=args.warmup_proportion,
        num_train_epochs=args.num_train_epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size
    )
    
    # create runner class for training and evaluation tasks.
    runner = GlueTaskClassifierRunner(
        encoder_model = model,
        classifier_model = classifier,
        optimizer = optimizer,
        label_list = task.get_labels(),
        device = device,
        rparams = r_params
    )


    if args.do_train:
        runner.run_train_classifier(train_examples)

    if args.do_val:
        val_examples = task.get_dev_examples()
        results = runner.run_val() # TODO...


        # HACK for MNLI-mismatched
        if task.name == "mnli":
            mm_val_example = MnliMismatchedProcessor().get_dev_examples(task.data_dir)
            mm_results = runner.run_val()# TODO...






if __name__ == "__main__":
    main()