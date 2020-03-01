# Glue Task Runner Classes.

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


import logging
from tqdm import tqdm, trange
from .tasks import InputExample, InputFeatures


logger = logging.getLogger(__name__)

def is_null_label_map(label_map):
    return len(label_map) == 1 and label_map[None] == 0


def get_label_mode(label_map):
    if is_null_label_map(label_map):
        return LabelModes.REGRESSION
    else:
        return LabelModes.CLASSIFICATION


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


def convert_example_to_feature(example, label_map):
    '''convert example to feature. only change the label into feature if label map exists'''
    s1, s2 = example.text_a, example.text_b
    if is_null_label_map(label_map):
        label_id = example.label
    else:
        label_id = label_map[example.label]
    return InputFeatures(
        guid=example.guid,
        sent1=s1,
        sent2=s2,
        label_id=label_id
    )


def convert_examples_to_features(examples, label_map, verbose=True):
    """Load a dataset into a list of `InputFeatures`."""
    features = []
    for (ex_index, example) in enumerate(examples):
        feature_instance = convert_example_to_feature(example=example, label_map=label_map)
        if verbose and ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("sentence 1: %s" % example.text_a)
            logger.info("sentence 2: %s" % example.text_b)
            logger.info("label: %s (id = %d)" % (example.label, feature_instance.label_id))
        features.append(feature_instance)
    return features


def get_full_batch(features, label_mode):
    full_batch = features_to_data(features, label_mode=label_mode)
    return full_batch



def features_to_data(features, label_mode):
    if label_mode == LabelModes.CLASSIFICATION:
        label_type = torch.long
    elif label_mode == LabelModes.REGRESSION:
        label_type = torch.float
    else:
        raise KeyError(label_mode)
    return Batch(
        sent1s=[f.sent1 for f in features],
        sent2s=[f.sent2 for f in features],
        label_ids=torch.tensor([f.label_id for f in features], dtype=label_type),
    )



class LabelModes:
    CLASSIFICATION = "CLASSIFICATION"
    REGRESSION = "REGRESSION"


class Batch:
    def __init__(self, sent1s, sent2s, label_ids):
        self.sent1s = sent1s
        self.sent2s = sent2s
        self.label_ids = label_ids

    def to(self, device):
        return Batch(
            sent1s=self.sent1s.to(device),
            sent2s=self.sent2s.to(device),
            label_ids=self.label_ids.to(device),
        )

    def __len__(self):
        return len(self.sent1s)

    def __getitem__(self, key):
        return Batch(
            sent1s=self.sent1s[key],
            sent2s=self.sent2s[key],
            label_ids=self.label_ids[key],
        )


class TrainEpochState:
    def __init__(self):
        self.tr_loss = 0
        self.global_step = 0
        self.nb_tr_examples = 0
        self.nb_tr_steps = 0


class RunnerParameters:
    def __init__(self, local_rank, n_gpu,learning_rate, gradient_accumulation_steps, 
                t_total, warmup_proportion,
                num_train_epochs, train_batch_size, eval_batch_size):
        self.local_rank = local_rank
        self.n_gpu = n_gpu
        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.t_total = t_total
        self.warmup_proportion = warmup_proportion
        self.num_train_epochs = num_train_epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size



class GlueTaskClassifierRunner:

    def __init__(self, encoder_model, classifier_model, optimizer, label_list, device, rparams):
        self.encoder_model = encoder_model
        self.classifier_model = classifier_model
        self.optimizer = optimizer
        self.label_list = label_list
        self.label_map = {v: i for i, v in enumerate(label_list)}
        self.device = device
        self.rparams = rparams
    
    def run_train_classifier(self, train_examples, verbose=True):
        if verbose:
            logger.info("***** Running Training for Classifier *****")
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", self.rparams.train_batch_size)
            logger.info("  Num steps = %d", self.rparams.t_total)
        train_dataloader = self.get_train_dataloader(train_examples, verbose=verbose)

        for _ in trange(int(self.rparams.num_train_epochs), desc="Epoch"):
            self.run_train_epoch(train_dataloader)

    def run_train_epoch(self, train_dataloader):
        for _ in self.run_train_epoch_context(train_dataloader):
            pass
    
    def run_train_epoch_context(self, train_dataloader):
        self.classifier_model.train()
        self.encoder_model.eval()
        train_epoch_state = TrainEpochState()
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            self.run_train_step(
                step=step,
                batch=batch,
                train_epoch_state=train_epoch_state,
            )
            yield step, batch, train_epoch_state

    def run_train_step(self, step, batch, train_epoch_state):

        
        self.encoder_model.eval()
        # need to convert to list of sentences
        batch = batch.to(self.device)
        # TODO ...

        s1 = []
        s2 = []

        # TODO ...

        with torch.no_grad():
            # sent1 sent2 embeddings...
            s1_emb = self.encoder_model.encode(s1, bsize=rparams.train_batch_size, tokenize=False, verbose=True)
            s2_emb = self.encoder_model.encode(s2, bsize=rparams.train_batch_size, tokenize=False, verbose=True
        
        loss = self.classifier_model(s1_emb, s2_emb, labels = batch.label_ids)
        if self.rparams.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if self.rparams.gradient_accumulation_steps > 1:
            loss = loss / self.rparams.gradient_accumulation_steps
        else:
            loss.backward()

        train_epoch_state.tr_loss += loss.item()
        train_epoch_state.nb_tr_examples += batch.input_ids.size(0)
        train_epoch_state.nb_tr_steps += 1
        if (step + 1) % self.rparams.gradient_accumulation_steps == 0:
            # modify learning rate with special warm up BERT uses
            lr_this_step = self.rparams.learning_rate * warmup_linear(
                train_epoch_state.global_step / self.rparams.t_total, self.rparams.warmup_proportion)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_this_step
            self.optimizer.step()
            self.optimizer.zero_grad()
            train_epoch_state.global_step += 1

    # TODO
    def run_val(self, val_examples, task_name, verbose=True):
        val_dataloader = self.get_eval_dataloader(val_examples, verbose=verbose)
        self.classifier_model.eval()
        self.bert_model.eval()

        total_eval_loss = 0
        nb_eval_steps, nb_eval_examples = 0, 0
        all_logits = []
        all_labels = []
        for step, batch in enumerate(tqdm(val_dataloader, desc="Evaluating (Val)")):
            batch = batch.to(self.device)

            with torch.no_grad():
                _, pooled_output = self.bert_model(batch.input_ids, batch.segment_ids, batch.input_mask, output_all_encoded_layers=False)
                tmp_eval_loss = self.classifier_model(pooled_output, batch.label_ids)
                logits = self.classifier_model(pooled_output)
                label_ids = batch.label_ids.cpu().numpy()

            logits = logits.detach().cpu().numpy()
            total_eval_loss += tmp_eval_loss.mean().item()

            nb_eval_examples += batch.input_ids.size(0)
            nb_eval_steps += 1
            all_logits.append(logits)
            all_labels.append(label_ids)
        eval_loss = total_eval_loss / nb_eval_steps
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        return {
            "logits": all_logits,
            "loss": eval_loss,
            "metrics": compute_task_metrics(task_name, all_logits, all_labels),
        }

    def get_train_dataloader(self, train_examples, verbose=True):
        train_features = convert_examples_to_features(
            train_examples, self.label_map, self.rparams.max_seq_length, self.tokenizer,
            verbose=verbose,
        )
        train_data, train_tokens = convert_to_dataset(
            train_features, label_mode=get_label_mode(self.label_map),
        )
        if self.rparams.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=self.rparams.train_batch_size,
        )
        return HybridLoader(train_dataloader, train_tokens)

    def get_eval_dataloader(self, eval_examples, verbose=True):
        eval_features = convert_examples_to_features(
            eval_examples, self.label_map, self.rparams.max_seq_length, self.tokenizer,
            verbose=verbose,
        )
        eval_data, eval_tokens = convert_to_dataset(
            eval_features, label_mode=get_label_mode(self.label_map),
        )
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=self.rparams.eval_batch_size,
        )
        return HybridLoader(eval_dataloader, eval_tokens)
    
