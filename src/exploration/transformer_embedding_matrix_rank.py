from typing import List
import argparse
import os
import torch
from datasets import load_dataset
from transformers import AlbertConfig, AlbertTokenizer, AlbertModel
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_random_long_text_input(dataset, tokenizer, min_length: int = 300) -> dict:
    while True:
        idx = torch.randint(len(dataset['train']), (1,)).item()
        text = dataset['train'][idx]['text']
        tokens = tokenizer(text, return_tensors='pt', truncation=True)
        if tokens['input_ids'].shape[1] > min_length:
            return tokens

def compute_matrix_ranks(embeddings: List[torch.Tensor]) -> List[int]:
    ranks = []
    for z in tqdm(embeddings):
        z = z.squeeze(0)  # [seq_len, hidden_dim]
        rank = torch.linalg.matrix_rank(z).item()
        ranks.append(rank)
    return ranks

def plot_matrix_ranks(ranks: List[int], save_path: str = None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ranks, marker='o', linewidth=2, color='#2ca02c')
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_title('Embedding Matrix Rank per Layer', fontsize=20)
    ax.set_xlabel('Layer', fontsize=18)
    ax.set_ylabel('Matrix Rank', fontsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout(pad=2)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--random-seed', type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.random_seed)

    # Load dataset and model.
    wikitext = load_dataset("wikitext", "wikitext-103-v1")
    tokenizer = AlbertTokenizer.from_pretrained('albert-xlarge-v2')
    config = AlbertConfig.from_pretrained("albert-xlarge-v2", num_hidden_layers=48, num_attention_heads=1)
    model = AlbertModel.from_pretrained("albert-xlarge-v2", config=config)

    # Run model on a random long input.
    tokens = get_random_long_text_input(wikitext, tokenizer)

    # Compute the matrix rank of the embeddings.
    with torch.no_grad():
        output = model(**tokens, output_hidden_states=True)
        ranks = compute_matrix_ranks(output.hidden_states)

    plot_matrix_ranks(ranks, save_path='../../visualization/transformer/embedding_matrix_rank_albert_xlarge_v2.png')
