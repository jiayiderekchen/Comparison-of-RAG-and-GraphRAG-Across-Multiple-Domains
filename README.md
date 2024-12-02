# Comparative Analysis of RAG and GraphRAG Retrievers for Knowledge Retrieval in Diverse Datasets

## Motivation 
This study presents a comparative analysis of retriever part of traditional Retrieval-Augmented Generation (RAG) system, and GraphRAG, a graph-enhanced RAG approach. By leveraging datasets across multiple domains, including SQuAD, PubMedQA, and arXiv, the research evaluates retrieval quality, efficiency, and the impact of graph structures on performance.

## Design of RAG and GraphRAG 
### BaseRAG
![BaseRAG](https://github.com/jiayiderekchen/Comparison-of-RAG-and-GraphRAG-Across-Multiple-Domains/blob/main/BaseRAG.png)
### GraphRAG
![GraphRAG](https://github.com/jiayiderekchen/Comparison-of-RAG-and-GraphRAG-Across-Multiple-Domains/blob/main/GraphRAG.png)

[**Code**](https://github.com/jiayiderekchen/Comparison-of-RAG-and-GraphRAG-Across-Multiple-Domains/blob/main/graphrag_baserag_code.py)

## Benchmark Datasets 
- [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)
- [PubMedQA](https://pubmedqa.github.io/)
- [arXiv](https://www.kaggle.com/datasets/Cornell-University/arxiv)

## Results

[**Report**](https://github.com/jiayiderekchen/Comparison-of-RAG-and-GraphRAG-Across-Multiple-Domains/blob/main/Comparative%20Analysis%20of%20RAG%20and%20GraphRAG%20Retrievers%20for%20Knowledge%20Retrieval%20in%20Diverse%20Datasets.pdf) 

## Conclusion 
The findings suggest that the choice between BaseRAG and GraphRAG depends on the application’s domain and requirements. For tasks involving complex relationships and smaller datasets, GraphRAG is a promising choice. Conversely, BaseRAG remains the preferred option for general or large-scale applications.
