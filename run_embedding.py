import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

import logging
import gc
import os

from tqdm.auto import tqdm, trange
from tasks import get_task, MnliMismatchedProcessor
from models import InferSent


logger = logging.getLogger(__name__)

from runner import (
    get_label_mode,
    convert_examples_to_features,
    get_full_batch,
)


def get_dataloader(input_examples, label_map):
    train_features = convert_examples_to_features(
        input_examples, label_map, verbose=True,
    )
    full_batch = get_full_batch(
        train_features, label_mode=get_label_mode(label_map),
    )
    sampler = SequentialSampler(full_batch.pairs)
    dataloader = DataLoader(
        full_batch.pairs, sampler=sampler, batch_size=config['bsize'],
    )
    return dataloader


def init_output_dir(path):
    os.makedirs(path, exist_ok=True)


# configuration
dataset_path = 'dataset/WNLI'
output_dir = 'savedir/WNLI'
word_emb_path = 'dataset/fastText/crawl-300d-2M.vec'
model_path = 'dataset/encoder/infersent2.pkl'
task_name = 'wnli'

config = {
    'word_emb_dim': 300,
    'enc_lstm_dim': 2048,
    'n_enc_layers': 1,
    'dpout_model': 0,
    'dpout_fc': 0,
    'fc_dim': 512,
    'bsize': 64,
    'n_classes': 3,
    'pool_type': 'max',
    'nonlinear_fc': 0,
    'use_cuda': True,
    'version': 2,
}


def run_encoding(loader, model, mode='train'):
    # run embedding for train set
    count = 0
    tensor_list_a, tensor_list_b, labels_tensor_list = [], [], []
    for step, batch in enumerate(tqdm(loader, desc="Encode {} set".format(mode))):
        s1 = list(batch[0])
        s2 = list(batch[1])
        with torch.no_grad():
            s1_emb = model.encode(s1, bsize=config['bsize'], tokenize=True, verbose=False)
            s2_emb = model.encode(s2, bsize=config['bsize'], tokenize=True, verbose=False)
        s1_emb = torch.tensor(s1_emb)
        s2_emb = torch.tensor(s2_emb)
        labels = batch[-1] if (mode != 'test' or mode != 'mm_test') else None

        tensor_list_a.append(s1_emb)
        tensor_list_b.append(s2_emb)
        if mode != 'test' or mode != 'mm_test':
            labels_tensor_list.append(labels)

        if (step + 1) % 2000 == 0:
            # save per every 2000 steps for solving not enough memory problem
            train_emb_a = torch.cat(tensor_list_a).cpu()
            tensor_list_a = []
            train_emb_b = torch.cat(tensor_list_b).cpu()
            tensor_list_b = []

            print("shape of {} set sentence a: {}".format(mode, train_emb_a.shape))
            print("shape of {} set sentence b: {}".format(mode, train_emb_b.shape))

            if mode != 'test' or mode != 'mm_test':
                train_labels = torch.cat(labels_tensor_list).cpu()
                labels_tensor_list = []
                print("shape of {} set labels: {}".format(mode, train_labels.shape))
                dataset_embeddings = TensorDataset(train_emb_a, train_emb_b, train_labels)
            else:
                dataset_embeddings = TensorDataset(train_emb_a, train_emb_b)

            # save to output dir
            torch.save(dataset_embeddings,
                       os.path.join(output_dir, "{}-{}.dataset".format(mode, count)))
            print("embeddings saved at: {}".format(os.path.join(output_dir, "{}-{}.dataset".format(mode, count))))
            del dataset_embeddings
            count += 1
    # save the rest part
    train_emb_a = torch.cat(tensor_list_a).cpu()
    train_emb_b = torch.cat(tensor_list_b).cpu()

    print("shape of {} set sentence a: {}".format(mode, train_emb_a.shape))
    print("shape of {} set sentence b: {}".format(mode, train_emb_b.shape))

    if mode != 'test' or 'mm_test':
        train_labels = torch.cat(labels_tensor_list).cpu()
        print("shape of {} set labels: {}".format(mode, train_labels.shape))
        dataset_embeddings = TensorDataset(train_emb_a, train_emb_b, train_labels)
    else:
        dataset_embeddings = TensorDataset(train_emb_a, train_emb_b)

    # save to output dir
    torch.save(dataset_embeddings,
               os.path.join(output_dir, "{}-{}.dataset".format(mode, count)))
    print("embeddings saved at: {}".format(os.path.join(output_dir, "{}-{}.dataset".format(mode, count))))
    del dataset_embeddings
    gc.collect()


def main():
    init_output_dir(output_dir)
    # prepare dataset
    task = get_task(task_name, dataset_path)
    label_list = task.get_labels()
    label_map = {v: i for i, v in enumerate(label_list)}

    print("loading raw data ... ")
    train_examples = task.get_train_examples()
    val_examples = task.get_dev_examples()
    test_examples = task.get_test_examples()

    print("converting to data loader ... ")
    train_loader = get_dataloader(train_examples, label_map)
    val_loader = get_dataloader(val_examples, label_map)
    test_loader = get_dataloader(test_examples, label_map)

    # load model
    print("loading model ... ")
    model = InferSent(config)
    model.load_state_dict(torch.load(model_path))
    model = model.cuda() if config['use_cuda'] else model
    model.set_w2v_path(word_emb_path)
    print("building model vocabs ... ")
    model.build_vocab_k_words(K=100000, verbose=True)

    # run embedding for train set
    print("Run embedding for train set")
    for _ in trange(1, desc="Epoch"):
        run_encoding(loader=train_loader,
                     model=model,
                     mode='train')

    print("Run embedding for dev set")
    for _ in trange(1, desc="Epoch"):
        run_encoding(loader=val_loader,
                     model=model,
                     mode='dev')

    print("Run embedding for test set")
    for _ in trange(1, desc="Epoch"):
        run_encoding(loader=test_loader,
                     model=model,
                     mode='test')

    # HACK FOR MNLI mis-matched
    if task_name == 'mnli':
        print("Run Embedding for MNLI Mis-Matched Datasets")
        print("loading raw data ... ")
        mm_val_example = MnliMismatchedProcessor().get_dev_examples(dataset_path)
        mm_test_examples = MnliMismatchedProcessor().get_test_examples(dataset_path)
        print("converting to data loader ... ")
        mm_val_loader = get_dataloader(mm_val_example, label_map)
        mm_test_loader = get_dataloader(mm_test_examples, label_map)

        print("Run embedding for mm_dev set")
        for _ in trange(1, desc="Epoch"):
            run_encoding(loader=mm_val_loader,
                         model=model,
                         mode='mm_dev')

        print("Run embedding for test set")
        for _ in trange(1, desc="Epoch"):
            run_encoding(loader=mm_test_loader,
                         model=model,
                         mode='mm_test')


if __name__ == '__main__':
    main()
