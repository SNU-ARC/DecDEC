import torch
from tqdm import tqdm
import os
import logging
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from datasets import load_dataset
import random
import numpy as np
import logging


def _get_wikitext2(split):
    assert split in ['train', 'validation', 'test'], f"Unknown split {split} for wikitext2"

    data = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split, trust_remote_code=True)
    return data['text']


def _get_ptb(split, slice_unk=True):
    assert split in ['train', 'validation', 'test'], f"Unknown split {split} for ptb"

    data = load_dataset('ptb_text_only', 'penn_treebank', split=split,
                        trust_remote_code=True)
    data_list = data['sentence']

    if slice_unk:
        data_list = [s.replace('<unk>', '< u n k >') for s in data_list]

    return data_list


def _get_c4(split):
    assert split in ['train', 'validation'], f"Unknown split {split} for c4"

    if split == 'train':
        data = load_dataset(
            'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train',
            trust_remote_code=True
        )
    else:
        assert split == 'validation'
        data = load_dataset(
            'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation',
            trust_remote_code=True
        )

    return data['text']


def _get_pileval(split):
    if split != 'validation':
        logging.warning(f"Pileval only has a validation split, but got split={split}. Using validation split.")
    data = load_dataset("mit-han-lab/pile-val-backup", split="validation", trust_remote_code=True)

    return data['text']


def _sample_and_tokenize(texts, tokenizer, seq_len, num_samples, seed=None):
    assert num_samples <= len(texts), \
        f"num_samples({num_samples}) should be less than or equal to the number of texts({len(texts)})"

    # this works for None too, effectively setting random seeds
    random.seed(seed)
    np.random.seed(seed)

    selected_indices = set()

    samples = []
    while len(samples) < num_samples:
        idx = random.randint(0, len(texts) - 1)
        if idx in selected_indices:  # we don't want to sample the same text twice
            continue
        text = texts[idx]

        tokens = tokenizer(text, return_tensors='pt')['input_ids'][0]
        if len(tokens) < seq_len:  # if the text is too short, we skip it
            continue

        tokens = tokens[:seq_len]

        selected_indices.add(idx)
        samples.append(tokens)

    return samples


def _get_dataset(dataset_name, split):
    if dataset_name == 'wikitext2':
        return _get_wikitext2(split)
    elif dataset_name == 'ptb':
        return _get_ptb(split)
    elif dataset_name == 'c4':
        return _get_c4(split)
    elif dataset_name == 'pileval':
        return _get_pileval(split)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")


def get_tokens(dataset_name, split, tokenizer, seq_len, num_samples, seed=None):
    logging.info(f"Fetching dataset: {dataset_name}")
    texts = _get_dataset(dataset_name, split)
    logging.info(f"Sampling {num_samples} samples of length {seq_len} from {dataset_name}...")
    return _sample_and_tokenize(texts, tokenizer, seq_len, num_samples, seed)


def get_modules(layer):
    """Return all torch.nn.Linear modules within the given layer."""
    modules = {}
    for name, module in layer.named_modules():
        if isinstance(module, torch.nn.Linear):
            modules[name] = module
    return modules


def compute_thresholds(
        model_path,
        save_path,
        dataset='c4',
        seq_len=512,
        num_examples=100,
        random_state=0,
        cpu_only=False,
):
    # If save_path doesn't exist, create it
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    else:
        # Check that save_path is a directory
        if not os.path.isdir(save_path):
            raise ValueError(f"Provided save_path {save_path} is not a directory.")

    save_path = os.path.join(save_path, 'thresholds.pt')

    # Check if the file already exists
    if os.path.exists(save_path):
         input(f"[WARNING] File {save_path} already exists. Press enter to overwrite or Ctrl+C to cancel.")

    logging.info(f"Collecting thresholds on dataset {dataset} with sequence length {seq_len} and "
                 f"{num_examples} examples...")

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                 device_map='auto' if not cpu_only else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    input_tokens = get_tokens(dataset, 'train', tokenizer, seq_len, num_examples, seed=random_state)

    model.eval()

    layers = model.model.layers

    thresholds = [{} for _ in range(len(layers))]

    for l_idx, layer in enumerate(layers):
        for module_name, module in get_modules(layer).items():
            thresholds[l_idx][module_name] = torch.zeros(module.in_features, device=model.device)

    num_total_activation_samples = seq_len * num_examples

    def input_hook(module, inputs, outputs):
        activations = inputs[0]
        assert activations.dim() == 3 and activations.shape[0] == 1
        activations = activations.squeeze(0)

        # Now activations is a 2D tensor of shape (seq_len, hidden_size)
        sorted_abs_activations = torch.sort(torch.abs(activations), dim=1, descending=True).values

        # Get the max value for each column
        max_abs_activation_per_position = torch.max(sorted_abs_activations, dim=0).values

        # Store the results
        thresholds[module.layer_index][module.module_name] = \
            torch.max(thresholds[module.layer_index][module.module_name], max_abs_activation_per_position)

    hooks = []

    for l_idx, layer in enumerate(layers):
        for module_name, module in get_modules(layer).items():
            # Store layer and module name for identification
            module.layer_index = l_idx
            module.module_name = module_name
            # Register hook directly on the module
            hooks.append(module.register_forward_hook(input_hook))

    # Calculate gradients through loss.backward()
    with torch.no_grad():
        for tokens in tqdm(input_tokens, desc="Computing thresholds"):
            tokens = tokens.to(model.device)
            tokens = tokens.unsqueeze(0)
            model(input_ids=tokens)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Move model back to cpu
    model.cpu()

    # Move thresholds to cpu
    for l_idx, layer in enumerate(layers):
        for module_name, module in get_modules(layer).items():
            thresholds[l_idx][module_name] = thresholds[l_idx][module_name].cpu()

    # Save the thresholds to file
    logging.info(f"Saving thresholds to {save_path}...")
    torch.save(thresholds, save_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Compute activation thresholds for Linear modules in a model.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the thresholds file')
    parser.add_argument('--dataset', type=str, default='c4', choices=['wikitext2', 'ptb', 'c4', 'pileval'],
                        help='Dataset to use for collecting activations')
    parser.add_argument('--seq_len', type=int, default=512, help='Sequence length for each input sample')
    parser.add_argument('--num_examples', type=int, default=100, help='Number of examples to sample')
    parser.add_argument('--random_state', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--cpu_only', action='store_true',
                        help='If set, compute on CPU only. Useful if GPU memory is limited.')

    args = parser.parse_args()

    compute_thresholds(
        model_path=args.model_path,
        save_path=args.save_path,
        dataset=args.dataset,
        seq_len=args.seq_len,
        num_examples=args.num_examples,
        random_state=args.random_state,
        cpu_only=args.cpu_only,
    )