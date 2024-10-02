# Planning on Graphs (PoG)
Implementation of "Planning on Graphs: Enhancing Large Language Model Reasoning with Embedding-Based Pathfinding Algorithms"

Planning on Graphs (PoG) enhances the reasoning capabilities of large language models (LLMs) on knowledge graphs (KGs) through the integration of pathfinding algorithms. Our approach addresses the limitations of previous methods that required extensive fine-tuning of LLMs or demanded frequent interactions between the LLM and the KG. By introducing a planning-searching-reasoning framework, we leverage a semantic embedding model to facilitate more efficient interactions between the LLM and the KG. Additionally, we integrate semantic similarity as a cost metric within pathfinding algorithms, effectively combining these algorithms with KG reasoning.
## Requirements
To install the required dependencies, run:
```
pip install -r requirements.txt
```

## Datasets

> Datasets will be automatically downloaded from the open-source HuggingFace page.

## Inference
Run PoG with either Dijkstra's or the A* pathfinding algorithm.
### Method1: Planning with Dijkstra's algorithm

To execute the PoG with Dijkstra's algorithm, run:

```bash
#Fill in your OpenAI api_key in PoG_Dijkstra.py and utils.py
python PoG_Dijkstra.py \
        --dataset webqsp(or cwq) \ # choose your dataset
        --max_length 256 \ 
        --LLM_type gpt-4-turbo \ # the LLM you choose
        --prune_tools llm \
```

### Method2: Planning with A* algorithm

To execute the PoG with the A* algorithm, run:

```bash
#Fill in your OpenAI api_key in PoG_A_star.py and utils.py
python PoG_A_star.py \
        --dataset webqsp(or cwq) \ # choose your dataset
        --max_length 256 \ 
        --LLM_type gpt-4-turbo \ # the LLM you choose
        --prune_tools llm \
```

Answers will be saved at: `PoG/predictions/{dataset}`

### Evaluation Results

> The results will be evaluated automatically once the PoG process is complete.
