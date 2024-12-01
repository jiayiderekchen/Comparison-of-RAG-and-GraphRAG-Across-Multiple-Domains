# -*- coding: utf-8 -*-

# !pip install datasets
# !pip install datasets sentence-transformers networkx

import os
import json
import requests

def download_squad(dataset_type="train", save_path="squad"):
    """
    Download the SQuAD dataset.

    Args:
        dataset_type (str): "train" or "dev" (validation set).
        save_path (str): Directory to save the dataset.

    Returns:
        str: Path to the downloaded dataset file.
    """
    base_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
    file_name = f"{dataset_type}-v2.0.json"
    url = base_url + file_name

    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # File path to save
    file_path = os.path.join(save_path, file_name)

    # Download the file
    if not os.path.exists(file_path):
        print(f"Downloading {dataset_type} dataset...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"{dataset_type} dataset downloaded successfully and saved to {file_path}")
        else:
            raise Exception(f"Failed to download {dataset_type} dataset. HTTP Status: {response.status_code}")
    else:
        print(f"{dataset_type} dataset already exists at {file_path}")

    return file_path

def load_squad_data(filepath):
    """
    Load the SQuAD dataset and prepare documents, queries, and relevant documents.

    Args:
        filepath (str): Path to the SQuAD JSON file.

    Returns:
        tuple: documents, queries, relevant_docs
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    documents = []
    queries = []
    relevant_docs = []

    for topic in data['data']:
        for paragraph in topic['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                queries.append(qa['question'])
                relevant_docs.append(context)  # Map query to its context
                documents.append(context)  # Repeat document for each query

    return documents, queries, relevant_docs


# Example Usage
if __name__ == "__main__":
    # Step 1: Download the SQuAD train dataset
    squad_train_path = download_squad(dataset_type="train")

    # Step 2: Load the dataset
    documents_squad, queries_squad, relevant_docs_squad = load_squad_data(squad_train_path)

    # Step 3: Print some examples to verify
    print(f"Number of documents: {len(documents_squad)}")
    print(f"Number of queries: {len(queries_squad)}")
    print(f"Number of relevant docs: {len(relevant_docs_squad)}")
    print("\nSample Query and Relevant Doc:")
    print(f"Query: {queries_squad[0]}")
    print(f"Relevant Doc: {relevant_docs_squad[0]}")

import os
import requests
import json

def download_pubmedqa_from_github(save_path="pubmedqa"):
    """
    Download PubMedQA dataset files from GitHub.

    Args:
        save_path (str): Directory to save the dataset.

    Returns:
        tuple: Paths to the downloaded dataset files.
    """
    base_url = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/"
    files = ["ori_pqal.json", "test_ground_truth.json"]
    os.makedirs(save_path, exist_ok=True)

    downloaded_files = []
    for file_name in files:
        url = base_url + file_name
        file_path = os.path.join(save_path, file_name)

        # Download the file
        if not os.path.exists(file_path):
            print(f"Downloading {file_name}...")
            response = requests.get(url)
            if response.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(response.content)
                print(f"{file_name} downloaded successfully and saved to {file_path}")
            else:
                raise Exception(f"Failed to download {file_name}. HTTP Status: {response.status_code}")
        else:
            print(f"{file_name} already exists at {file_path}")

        downloaded_files.append(file_path)

    return downloaded_files


def load_pubmedqa_data(filepath):
    """
    Load PubMedQA dataset from a JSON file.

    Args:
        filepath (str): Path to the PubMedQA JSON file.

    Returns:
        tuple: documents, queries, relevant_docs
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    contexts = []
    long_answers = []
    queries = []
    relevant_docs = []

    for id_, entry in data.items():
        question = entry.get('QUESTION', '')
        context = " ".join(entry.get('CONTEXTS', []))
        long_answer = entry.get('LONG_ANSWER', '')

        if question:
            # Add contexts and long answers separately
            contexts.append(context)
            long_answers.append(long_answer)
            queries.append(question)
            relevant_docs.append(long_answer)  # Use long answer as the relevant document

    # Combine contexts and long answers into one retrieval pool
    documents = contexts + long_answers
    return documents, queries, relevant_docs



# Example Usage
if __name__ == "__main__":
    # Step 1: Download the PubMedQA dataset
    pubmedqa_files = download_pubmedqa_from_github()

    # Step 2: Load the dataset
    ori_pqal_path = pubmedqa_files[0]
    documents_pubmedqa, queries_pubmedqa, relevant_docs_pubmedqa = load_pubmedqa_data(ori_pqal_path)

def load_arxiv_data(filepath):
    """
    Load and preprocess the arXiv dataset.

    Args:
        filepath (str): Path to the JSON dataset file.

    Returns:
        tuple: documents, queries, relevant_docs
    """
    documents, queries, relevant_docs = [], [], []

    with open(filepath, 'r') as f:
        try:
            # Load the entire JSON array
            records = json.load(f)
            for record in records:
                documents.append(record.get("abstract", ""))
                queries.append(record.get("title", ""))
                relevant_docs.append(record.get("abstract", ""))
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return [], [], []

    return documents, queries, relevant_docs

from typing import List
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from community import community_louvain  # For Louvain community detection
import numpy as np
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import time

@dataclass
class RetrievalResult:
    """Data class to store retrieval results"""
    retrieved_docs: List[str]
    relevance_scores: List[float]
    retrieval_time: float

def encode_with_batches(encoder, texts, batch_size=32) -> np.ndarray:
    """
    Encode texts in batches for efficiency.
    """
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings.extend(encoder.encode(batch, convert_to_tensor=False))
    return np.vstack(embeddings)

class BaseRAG:
    """Base RAG implementation using simple vector similarity"""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None

    def add_documents(self, documents: List[str], batch_size: int = 32) -> None:
        self.documents = documents
        self.embeddings = encode_with_batches(self.encoder, documents, batch_size=batch_size)

    def retrieve(self, query: str, k: int = 3) -> RetrievalResult:
        start_time = time.time()
        query_embedding = self.encoder.encode(query)
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            self.embeddings
        )[0]
        top_indices = np.argsort(similarities)[-k:][::-1]
        retrieval_time = time.time() - start_time
        return RetrievalResult(
            retrieved_docs=[self.documents[i] for i in top_indices],
            relevance_scores=similarities[top_indices].tolist(),
            retrieval_time=retrieval_time
        )

class GraphRAG(BaseRAG):
    """Graph-enhanced RAG implementation with k-NN graph construction and community detection."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        super().__init__(model_name)
        self.graph = nx.Graph()
        self.node_communities = {}

    def add_documents(self, documents: List[str], batch_size: int = 32, k_neighbors: int = 10, similarity_threshold: float = 0.3):
        """
        Add documents to the retriever and construct a k-NN graph.
        """
        print("Encoding documents...")
        super().add_documents(documents, batch_size=batch_size)

        similarity_matrix = cosine_similarity(self.embeddings)
        print("Building graph...")

        for i in range(len(documents)):
            self.graph.add_node(i, content=documents[i])
            neighbors = np.argsort(similarity_matrix[i])[-(k_neighbors + 1):-1][::-1]
            for j in neighbors:
                if similarity_matrix[i, j] > similarity_threshold:
                    self.graph.add_edge(i, j, weight=similarity_matrix[i, j])

        print("Detecting communities...")
        self.node_communities = community_louvain.best_partition(self.graph, weight='weight')

        density = nx.density(self.graph)
        num_components = nx.number_connected_components(self.graph)
        num_communities = len(set(self.node_communities.values()))
        print(f"Graph constructed with density: {density:.4f}, {num_components} connected components, "
              f"and {num_communities} communities.")

    def retrieve(self, query: str, k: int = 10, sim_weight: float = 0.7, cluster_weight: float = 0.1) -> RetrievalResult:
        """
        Retrieve top-k documents based on similarity, PageRank, and community information.
        """
        start_time = time.time()
        query_embedding = self.encoder.encode(query)
        similarities = cosine_similarity(query_embedding.reshape(1, -1), self.embeddings)[0]

        pagerank_scores = nx.pagerank(self.graph)
        query_top_index = np.argmax(similarities)
        query_cluster = self.node_communities.get(query_top_index, -1)

        final_scores = []
        cluster_size = sum(1 for c in self.node_communities.values() if c == query_cluster)
        for i in range(len(self.documents)):
            cluster_bonus = cluster_weight / cluster_size if self.node_communities.get(i) == query_cluster else 0.0
            score = sim_weight * similarities[i] + (1 - sim_weight) * pagerank_scores[i] + cluster_bonus
            final_scores.append(score)

        final_scores = np.array(final_scores)
        top_indices = np.argsort(final_scores)[-k:][::-1]
        retrieval_time = time.time() - start_time
        return RetrievalResult(
            retrieved_docs=[self.documents[i] for i in top_indices],
            relevance_scores=final_scores[top_indices].tolist(),
            retrieval_time=retrieval_time
        )

# all datasets

def evaluate_retrievers(documents, queries, relevant_docs, k):
    """
    Evaluate BaseRAG and GraphRAG retrievers on the given dataset.

    Args:
        documents (List[str]): List of document texts.
        queries (List[str]): List of query texts.
        relevant_docs (List[str]): List of relevant documents for each query.
        k (int): Number of top documents to retrieve.

    Returns:
        Dict[str, Dict[str, float]]: Evaluation metrics for BaseRAG and GraphRAG.
    """
    # Initialize retrievers
    base_rag = BaseRAG()
    graph_rag = GraphRAG()

    # base_rag = BaseRAG(model_name="multi-qa-mpnet-base-dot-v1")
    # graph_rag = GraphRAG(model_name="multi-qa-mpnet-base-dot-v1")

    # Add documents to retrievers
    base_rag.add_documents(documents)
    graph_rag.add_documents(documents)

    # Initialize metrics
    metrics = {
        'BaseRAG': {'hits': 0, 'mrr': 0.0, 'precision': 0.0, 'recall': 0.0, 'avg_time': 0.0},
        'GraphRAG': {'hits': 0, 'mrr': 0.0, 'precision': 0.0, 'recall': 0.0, 'avg_time': 0.0}
    }

    # Evaluate queries
    for query, relevant in zip(queries, relevant_docs):
        relevant_set = {relevant}

        for retriever_name, retriever in [('BaseRAG', base_rag), ('GraphRAG', graph_rag)]:
            # Retrieve results
            result = retriever.retrieve(query, k)
            retrieved_docs = result.retrieved_docs[:k]
            retrieved_set = set(retrieved_docs)

            # Calculate metrics
            hits = int(bool(relevant_set.intersection(retrieved_set)))
            rank = (
                retrieved_docs.index(next(iter(relevant_set.intersection(retrieved_set)))) + 1
                if relevant_set.intersection(retrieved_set)
                else 0
            )
            precision = len(relevant_set.intersection(retrieved_set)) / k
            # recall = len(relevant_set.intersection(retrieved_set)) / len(relevant_set)  # Corrected
            recall = len(relevant_set.intersection(retrieved_set)) / len(relevant_set) if len(relevant_set) > 0 else 0

            # Update metrics
            metrics[retriever_name]['hits'] += hits
            metrics[retriever_name]['mrr'] += 1.0 / rank if rank else 0.0
            metrics[retriever_name]['precision'] += precision
            metrics[retriever_name]['recall'] += recall
            metrics[retriever_name]['avg_time'] += result.retrieval_time

    # Normalize metrics
    num_queries = len(queries)
    for retriever_name in metrics:
        for metric in metrics[retriever_name]:
            metrics[retriever_name][metric] /= num_queries

    return metrics


def reduce_dataset(documents, queries, relevant_docs, sample_size, dataset_name="pubmed"):
    """
    Reduce the size of the dataset for faster benchmarking.

    Args:
        documents (List[str]): List of all documents.
        queries (List[str]): List of all queries.
        relevant_docs (List[str]): List of all relevant documents.
        sample_size (int): Number of samples to take.
        dataset_name (str): Name of the dataset ("pubmed" or "squad").

    Returns:
        tuple: Reduced documents, queries, and relevant_docs.
    """
    print(f"Documents: {len(documents)}, Queries: {len(queries)}, Relevant Docs: {len(relevant_docs)}")

    if dataset_name == "pubmed":
        if len(queries) != len(relevant_docs):
            raise ValueError("Queries and relevant_docs lists must have the same size.")

        sample_size = min(sample_size, len(queries))  # Ensure we don't sample more than available
        indices = np.random.choice(len(queries), size=sample_size, replace=False)

        queries_subset = [queries[i] for i in indices]
        relevant_docs_subset = [relevant_docs[i] for i in indices]

        # Adjust the retrieval pool (documents) based on the selected subset
        retrieval_pool = [documents[i] for i in indices] + [documents[i + len(queries)] for i in indices]

        return retrieval_pool, queries_subset, relevant_docs_subset

    elif dataset_name == "squad":
        if not (len(documents) == len(queries) == len(relevant_docs)):
            raise ValueError("Documents, queries, and relevant_docs lists must have the same size.")

        sample_size = min(sample_size, len(documents), len(queries), len(relevant_docs))
        indices = list(range(len(queries)))
        np.random.shuffle(indices)  # Shuffle indices
        indices = indices[:sample_size]

        documents_subset = [documents[i] for i in indices]
        queries_subset = [queries[i] for i in indices]
        relevant_docs_subset = [relevant_docs[i] for i in indices]

        return documents_subset, queries_subset, relevant_docs_subset

    elif dataset_name == "arxiv":
        if len(queries) != len(relevant_docs):
            raise ValueError("Queries and relevant_docs lists must have the same size.")

        sample_size = min(sample_size, len(queries))  # Ensure we don't sample more than available
        indices = np.random.choice(len(queries), size=sample_size, replace=False)

        # Create subsets for arXiv dataset
        documents_subset = [documents[i] for i in indices]
        queries_subset = [queries[i] for i in indices]
        relevant_docs_subset = [relevant_docs[i] for i in indices]

        # print(f"Sampled Size: {len(documents_subset)}")  # Debug print
        return documents_subset, queries_subset, relevant_docs_subset


    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")



def run_benchmarks():
    """
    Run benchmarks on SQuAD and PubMedQA datasets.
    """
    # Load and evaluate SQuAD dataset
    print("Loading SQuAD dataset...")
    squad_path = "squad/train-v2.0.json"
    documents_squad, queries_squad, relevant_docs_squad = load_squad_data(squad_path)
    retrieval_pool, queries_subset, relevant_docs_subset = reduce_dataset(
        documents_squad, queries_squad, relevant_docs_squad, sample_size=3500, dataset_name="squad"
    )
    print("Running SQuAD benchmark...")
    metrics_squad = evaluate_retrievers(retrieval_pool, queries_subset, relevant_docs_subset, k=7)

    # Load and evaluate PubMedQA dataset
    print("Loading PubMedQA dataset...")
    pubmedqa_path = "pubmedqa/ori_pqal.json"
    documents_pubmedqa, queries_pubmedqa, relevant_docs_pubmedqa = load_pubmedqa_data(pubmedqa_path)
    retrieval_pool, queries_subset, relevant_docs_subset = reduce_dataset(
        documents_pubmedqa, queries_pubmedqa, relevant_docs_pubmedqa, sample_size=3500, dataset_name="pubmed"
    )
    print("Running PubMedQA benchmark...")
    metrics_pubmedqa = evaluate_retrievers(retrieval_pool, queries_subset, relevant_docs_subset, k=7)

    # Load and evaluate arXiv dataset
    print("Loading arXiv dataset...")
    arxiv_path = "/content/arxiv/arxiv-metadata-subset.json"
    documents, queries, relevant_docs = load_arxiv_data(arxiv_path)
    reduced_documents, reduced_queries, reduced_relevant_docs = reduce_dataset(
        documents, queries, relevant_docs, sample_size=3500, dataset_name="arxiv"
    )
    print("Running arXiv benchmark...")
    metrics_arxiv = evaluate_retrievers(reduced_documents, reduced_queries, reduced_relevant_docs, k=7)


    # Print results
    print("\nBenchmark Results:")
    for dataset_name, metrics in [
        ('SQuAD', metrics_squad),
        ('PubMedQA', metrics_pubmedqa),
        ('arXiv', metrics_arxiv)
    ]:
        print(f"\nDataset: {dataset_name}")
        for retriever_name, retriever_metrics in metrics.items():
            print(f"  {retriever_name}:")
            for metric, value in retriever_metrics.items():
                print(f"    {metric}: {value:.4f}")

run_benchmarks()

# single dataset

def evaluate_retrievers(documents, queries, relevant_docs, k):
    """
    Evaluate BaseRAG and GraphRAG retrievers on the given dataset.

    Args:
        documents (List[str]): List of document texts.
        queries (List[str]): List of query texts.
        relevant_docs (List[str]): List of relevant documents for each query.
        k (int): Number of top documents to retrieve.

    Returns:
        Dict[str, Dict[str, float]]: Evaluation metrics for BaseRAG and GraphRAG.
    """
    # Initialize retrievers
    # base_rag = BaseRAG()
    # graph_rag = GraphRAG()

    base_rag = BaseRAG(model_name="multi-qa-mpnet-base-dot-v1")
    graph_rag = GraphRAG(model_name="multi-qa-mpnet-base-dot-v1")

    # Add documents to retrievers
    base_rag.add_documents(documents)
    graph_rag.add_documents(documents)

    # Initialize metrics
    metrics = {
        'BaseRAG': {'hits': 0, 'mrr': 0.0, 'precision': 0.0, 'recall': 0.0, 'avg_time': 0.0},
        'GraphRAG': {'hits': 0, 'mrr': 0.0, 'precision': 0.0, 'recall': 0.0, 'avg_time': 0.0}
    }

    # Evaluate queries
    for query, relevant in zip(queries, relevant_docs):
        relevant_set = {relevant}

        for retriever_name, retriever in [('BaseRAG', base_rag), ('GraphRAG', graph_rag)]:
            # Retrieve results
            result = retriever.retrieve(query, k)
            retrieved_docs = result.retrieved_docs[:k]
            retrieved_set = set(retrieved_docs)

            # Calculate metrics
            hits = int(bool(relevant_set.intersection(retrieved_set)))
            rank = (
                retrieved_docs.index(next(iter(relevant_set.intersection(retrieved_set)))) + 1
                if relevant_set.intersection(retrieved_set)
                else 0
            )
            precision = len(relevant_set.intersection(retrieved_set)) / k
            # recall = len(relevant_set.intersection(retrieved_set)) / len(relevant_set)  # Corrected
            recall = len(relevant_set.intersection(retrieved_set)) / len(relevant_set) if len(relevant_set) > 0 else 0

            # Update metrics
            metrics[retriever_name]['hits'] += hits
            metrics[retriever_name]['mrr'] += 1.0 / rank if rank else 0.0
            metrics[retriever_name]['precision'] += precision
            metrics[retriever_name]['recall'] += recall
            metrics[retriever_name]['avg_time'] += result.retrieval_time

    # Normalize metrics
    num_queries = len(queries)
    for retriever_name in metrics:
        for metric in metrics[retriever_name]:
            metrics[retriever_name][metric] /= num_queries

    return metrics


def reduce_dataset(documents, queries, relevant_docs, sample_size, dataset_name="pubmed"):
    """
    Reduce the size of the dataset for faster benchmarking.

    Args:
        documents (List[str]): List of all documents.
        queries (List[str]): List of all queries.
        relevant_docs (List[str]): List of all relevant documents.
        sample_size (int): Number of samples to take.
        dataset_name (str): Name of the dataset ("pubmed" or "squad").

    Returns:
        tuple: Reduced documents, queries, and relevant_docs.
    """
    print(f"Documents: {len(documents)}, Queries: {len(queries)}, Relevant Docs: {len(relevant_docs)}")

    if dataset_name == "pubmed":
        if len(queries) != len(relevant_docs):
            raise ValueError("Queries and relevant_docs lists must have the same size.")

        sample_size = min(sample_size, len(queries))  # Ensure we don't sample more than available
        indices = np.random.choice(len(queries), size=sample_size, replace=False)

        queries_subset = [queries[i] for i in indices]
        relevant_docs_subset = [relevant_docs[i] for i in indices]

        # Adjust the retrieval pool (documents) based on the selected subset
        retrieval_pool = [documents[i] for i in indices] + [documents[i + len(queries)] for i in indices]

        return retrieval_pool, queries_subset, relevant_docs_subset

    elif dataset_name == "squad":
        if not (len(documents) == len(queries) == len(relevant_docs)):
            raise ValueError("Documents, queries, and relevant_docs lists must have the same size.")

        sample_size = min(sample_size, len(documents), len(queries), len(relevant_docs))
        indices = list(range(len(queries)))
        np.random.shuffle(indices)  # Shuffle indices
        indices = indices[:sample_size]

        documents_subset = [documents[i] for i in indices]
        queries_subset = [queries[i] for i in indices]
        relevant_docs_subset = [relevant_docs[i] for i in indices]

        return documents_subset, queries_subset, relevant_docs_subset

    elif dataset_name == "arxiv":
        if len(queries) != len(relevant_docs):
            raise ValueError("Queries and relevant_docs lists must have the same size.")

        sample_size = min(sample_size, len(queries))  # Ensure we don't sample more than available
        indices = np.random.choice(len(queries), size=sample_size, replace=False)

        # Create subsets for arXiv dataset
        documents_subset = [documents[i] for i in indices]
        queries_subset = [queries[i] for i in indices]
        relevant_docs_subset = [relevant_docs[i] for i in indices]

        # print(f"Sampled Size: {len(documents_subset)}")  # Debug print
        return documents_subset, queries_subset, relevant_docs_subset


    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

def run_benchmarks_single(dataset_name: str, sample_size: int, k: int):
    """
    Run benchmarks on the specified dataset.

    Args:
        dataset_name (str): The name of the dataset to run ("squad", "pubmed", "arxiv").
        sample_size (int): Number of samples to use from the dataset.
        k (int): Number of top documents to retrieve.

    Returns:
        None: Prints evaluation metrics for the specified dataset.
    """
    if dataset_name == "squad":
        # Load and evaluate SQuAD dataset
        print("Loading SQuAD dataset...")
        squad_path = "squad/train-v2.0.json"
        documents_squad, queries_squad, relevant_docs_squad = load_squad_data(squad_path)
        retrieval_pool, queries_subset, relevant_docs_subset = reduce_dataset(
            documents_squad, queries_squad, relevant_docs_squad, sample_size=sample_size, dataset_name="squad"
        )
        print(f"Running SQuAD benchmark with {sample_size} samples and k={k}...")
        metrics_squad = evaluate_retrievers(retrieval_pool, queries_subset, relevant_docs_subset, k=k)
        print("\nBenchmark Results for SQuAD:")
        for retriever_name, retriever_metrics in metrics_squad.items():
            print(f"  {retriever_name}:")
            for metric, value in retriever_metrics.items():
                print(f"    {metric}: {value:.4f}")

    elif dataset_name == "pubmed":
        # Load and evaluate PubMedQA dataset
        print("Loading PubMedQA dataset...")
        pubmedqa_path = "pubmedqa/ori_pqal.json"
        documents_pubmedqa, queries_pubmedqa, relevant_docs_pubmedqa = load_pubmedqa_data(pubmedqa_path)
        retrieval_pool, queries_subset, relevant_docs_subset = reduce_dataset(
            documents_pubmedqa, queries_pubmedqa, relevant_docs_pubmedqa, sample_size=sample_size, dataset_name="pubmed"
        )
        print(f"Running PubMedQA benchmark with {sample_size} samples and k={k}...")
        metrics_pubmedqa = evaluate_retrievers(retrieval_pool, queries_subset, relevant_docs_subset, k=k)
        print("\nBenchmark Results for PubMedQA:")
        for retriever_name, retriever_metrics in metrics_pubmedqa.items():
            print(f"  {retriever_name}:")
            for metric, value in retriever_metrics.items():
                print(f"    {metric}: {value:.4f}")

    elif dataset_name == "arxiv":
        # Load and evaluate arXiv dataset
        print("Loading arXiv dataset...")
        arxiv_path = "/content/arxiv/arxiv-metadata-subset.json"
        documents, queries, relevant_docs = load_arxiv_data(arxiv_path)
        reduced_documents, reduced_queries, reduced_relevant_docs = reduce_dataset(
            documents, queries, relevant_docs, sample_size=sample_size, dataset_name="arxiv"
        )
        print(f"Running arXiv benchmark with {sample_size} samples and k={k}...")
        metrics_arxiv = evaluate_retrievers(reduced_documents, reduced_queries, reduced_relevant_docs, k=k)
        print("\nBenchmark Results for arXiv:")
        for retriever_name, retriever_metrics in metrics_arxiv.items():
            print(f"  {retriever_name}:")
            for metric, value in retriever_metrics.items():
                print(f"    {metric}: {value:.4f}")

    else:
        print(f"Error: Unsupported dataset '{dataset_name}'. Please choose 'squad', 'pubmed', or 'arxiv'.")

run_benchmarks_single(dataset_name="arxiv", sample_size=10000, k=10)





