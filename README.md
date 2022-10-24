## Chemical-Gene-Chemical-Gene-Relation-Extraction-with-GNN

The model identifies chemical components and genes named entities and extracts the relations of the chemical-gene pair jointly. It utilizes the BioBERT model for word embedding and uses the GNN model for named recognition and RE subtasks. 

## Model Architecture

![Architecture2](https://user-images.githubusercontent.com/59030870/197599628-e47f1ec4-34a2-4aa0-ac7b-c5c11d9c5568.png)

## Model sections

- Word Embedding using BioBERT
- Graph Attention Networks
    -  Calculate Attention Weights
- Relation Extraction and Tagging





## Requirements

  - python 3.7
  - torch 1.3
  - tqdm
  - transformers
  - numpy

