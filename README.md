# Knowledge graph Assisted Reasoning Path Aggregation (KARPA)
Implementation of "KARPA: A Training-free Method of Adapting Knowledge Graph as References for Large Language Model's Reasoning Path Aggregation".

This repository contains the implementation of Knowledge graph Assisted Reasoning Path Aggregation (KARPA), a training-free framework designed for Knowledge Graph Question Answering (KGQA). Our method leverages the global planning capabilities of large language models (LLMs) to retrieve reasoning paths over knowledge graphs (KGs). KARPA addresses the limitations of stepwise LLM-based KGQA methods by enabling LLMs to generate initial, globally planned relation paths that represent reasoning chains from a topic entity to potential answer entities. Our method effectively reduces the number of interactions between the LLM and the KG, improving both efficiency and performance in KGQA tasks. Compared to existing methods, KARPA significantly enhances the accuracy of reasoning over KGs while minimizing computational costs.

## Requirements
To install the required dependencies, run:
```
pip install -r requirements.txt
```

## Datasets and Embedding Model

> Datasets and the embedding model will be automatically downloaded from the open-source HuggingFace pages.

## Run KARPA with different LLMs and retrieval methods
KARPA provides the implementations with different LLMs and retrieval methods.
### KARPA with heuristic value-based retrieval method

To execute the KARPA-H (KARPA with heuristic value-based retrieval method) with different LLMs, run:

```bash
cd KARPA
python KARPA_H.py \
        --dataset webqsp / cwq \ # choose the dataset
        --opeani_api_keys <your_api_key> \ # add the api key for the LLM
        --LLM_type gpt-4o-mini \ # the LLM you choose (e.g., gpt-4o, gpt-4o-mini)
        --LLM_URL <optional_LLM_API_URL> \ # add the URL if needed for API key
```

### KARPA with pathfinding-based retrieval method

To execute the KARPA-D (KARPA with pathfinding-based retrieval method) with different LLMs, run:

```bash
cd KARPA
python KARPA_D.py \
        --dataset webqsp / cwq \ # choose the dataset
        --opeani_api_keys <your_api_key> \ # add the api key for the LLM
        --LLM_type gpt-4o-mini \ # the LLM you choose (e.g., gpt-4o, gpt-4o-mini)
        --LLM_URL <optional_LLM_API_URL> \ # add the URL if needed for API key
```

Answers will be saved at: `predictions/{dataset}/{retrieval_method}`

### Evaluation Results

> The results will be evaluated automatically after the KARPA process completes. The evaluated results will also be saved in `predictions/{dataset}/{retrieval_method}`. Additionally, logs will be available for further review.
