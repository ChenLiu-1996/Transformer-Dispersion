from typing import List
import argparse
import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import phate
from datasets import load_dataset
from transformers import AlbertTokenizer, AlbertConfig, AlbertModel


def get_random_long_text_input(dataset, tokenizer, min_length: int = 300) -> dict:
    while True:
        idx = torch.randint(len(dataset['train']), (1,)).item()
        text = dataset['train'][idx]['text']
        tokens = tokenizer(text, return_tensors='pt', truncation=True)
        if tokens['input_ids'].shape[1] > min_length:
            return tokens


def auto_kmeans(z_2d: np.ndarray, max_k: int = 10, random_state: int = 42):
    best_k = 2
    best_score = -1
    best_labels = None

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=random_state)
        labels = kmeans.fit_predict(z_2d)

        if len(np.unique(labels)) == 1:
            continue  # skip trivial clustering

        score = silhouette_score(z_2d, labels)
        if score > best_score:
            best_k = k
            best_score = score
            best_labels = labels

    return best_k, best_labels


def compute_embedding_clusters(embeddings: List[torch.Tensor],
                               step: int = 10,
                               max_k: int = 10,
                               method: str = 'phate',
                               perplexity: int = 30,
                               random_seed: int = 42) -> List[dict]:
    selected = [(i, emb) for i, emb in enumerate(embeddings) if i % step == 0]
    layer_cluster_data = []

    for layer_idx, emb in selected:
        z = torch.nn.functional.normalize(emb.squeeze(0), dim=1).cpu().numpy()  # [seq_len, dim]
        if method == 'phate':
            dim_reduction_op = phate.PHATE(n_components=2, random_state=random_seed)
        elif method == 'tsne':
            dim_reduction_op = TSNE(n_components=2, perplexity=perplexity, random_state=random_seed)

        z_2d = dim_reduction_op.fit_transform(z)

        best_k, labels = auto_kmeans(z_2d, max_k=max_k, random_state=random_seed)

        layer_cluster_data.append({'layer': layer_idx, 'points': z_2d, 'labels': labels})

    return layer_cluster_data


def plot_embedding_cluster(cluster_data: List[dict], save_path: str = None, method: str = 'phate'):
    num_plots = len(cluster_data)
    cols = math.ceil(math.sqrt(num_plots))
    rows = math.ceil(num_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()

    for ax, data in zip(axes, cluster_data):
        points, labels = data['points'], data['labels']
        for cluster_id in np.unique(labels):
            cluster_points = points[labels == cluster_id]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], s=10, label=f'C{cluster_id}', alpha=0.7)
        ax.set_title(f"Layer {data['layer']} ({len(np.unique(labels))} clusters)", fontsize=18)
        ax.axis('off')

    for ax in axes[num_plots:]:
        ax.axis('off')

    if method == 'phate':
        method_name = 'PHATE'
    elif method == 'tsne':
        method_name = 't-SNE'

    fig.suptitle(f'{method_name} Embedding Clusters per Layer', fontsize=24)
    fig.tight_layout(rect=[0, 0, 1, 0.96], pad=2)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--random-seed', type=int, default=1)
    parser.add_argument('--method', type=str, default='phate')
    args = parser.parse_args()

    torch.manual_seed(args.random_seed)

    # Load dataset and model.
    wikitext = load_dataset("wikitext", "wikitext-103-v1")
    tokenizer = AlbertTokenizer.from_pretrained('albert-xlarge-v2')
    config = AlbertConfig.from_pretrained("albert-xlarge-v2", num_hidden_layers=48, num_attention_heads=1)
    model = AlbertModel.from_pretrained("albert-xlarge-v2", config=config)

    # Run model on a random long input.
    tokens = get_random_long_text_input(wikitext, tokenizer)

    # Dimensionality-reduce and plot the embeddings in 2D.
    with torch.no_grad():
        output = model(**tokens, output_hidden_states=True)
        cluster_data = compute_embedding_clusters(
            embeddings=output.hidden_states,
            step=2,
            method=args.method,
            max_k=30,
            perplexity=5,
            random_seed=args.random_seed)

    plot_embedding_cluster(
        cluster_data,
        save_path=f'../visualization/embedding_clusters_{args.method}_albert_xlarge_v2.png',
        method=args.method)
