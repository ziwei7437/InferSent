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
import json
import pandas as pd

import logging

import numpy as np

import torch
from torch import optim
import torch.nn as nn

from tasks import get_task, MnliMismatchedProcessor
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

    # others
    parser.add_argument("--verbose", action="store_true", help='showing information.')
    
    args = parser.parse_args(*in_args)
    return args


def print_args(args):
    for k, v in vars(args).items():
        print("  {}: {}".format(k, v))


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
    use_cuda = False if args.no_cuda else True
    verbose = args.verbose


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
        'use_cuda'       :  use_cuda            ,
        'version'        :  args.model_version  ,
        'dropout_prob'   :  args.dropout_prob   ,
    }

    # load model
    if verbose:
        print('loading model...')
    model = InferSent(config)
    model.load_state_dict(torch.load(args.model_path))
    model = model.cuda() if not args.no_cuda else model
    model.set_w2v_path(args.word_emb_path)
    model.build_vocab_k_words(k=args.k_freq_words, verbose=verbose)

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
        eval_batch_size=args.eval_batch_size,
        verbose=verbose
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
        results = runner.run_val(val_examples, task_name=task.name, verbose=verbose)

        df = pd.DataFrame(results["logits"])
        df.to_csv(os.path.join(args.output_dir, "val_preds.csv"), header=False, index=False)
        metrics_str = json.dumps({"loss": results["loss"], "metrics": results["metrics"]}, indent=2)
        print(metrics_str)
        with open(os.path.join(args.output_dir, "val_metrics.json"), "w") as f:
            f.write(metrics_str)

        # HACK for MNLI-mismatched
        if task.name == "mnli":
            mm_val_example = MnliMismatchedProcessor().get_dev_examples(task.data_dir)
            mm_results = runner.run_val(mm_val_example, task_name=task.name, verbose=verbose)

            df = pd.DataFrame(results["logits"])
            df.to_csv(os.path.join(args.output_dir, "mm_val_preds.csv"), header=False, index=False)
            combined_metrics = {}
            for k, v in results["metrics"].items():
                combined_metrics[k] = v
            for k, v in mm_results["metrics"].items():
                combined_metrics["mm-"+k] = v
            combined_metrics_str = json.dumps({
                "loss": results["loss"],
                "metrics": combined_metrics,
            }, indent=2)
            print(combined_metrics_str)
            with open(os.path.join(args.output_dir, "val_metrics.json"), "w") as f:
                f.write(combined_metrics_str)






if __name__ == "__main__":
    main()