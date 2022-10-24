## Chemical-Gene-Relation-Extraction-with-GNN

The model identifies chemical components and genes named entities and extracts the relations of the chemical-gene pair jointly. It utilizes the BioBERT model for word embedding and uses the GNN model for named recognition and RE subtasks. 

## Model Architecture

- Word Embedding using BioBERT
- Graph Attention Networks
    -  Calculate Attention Weights
    -  Update Vector Representations
- Relation Extraction and Tagging
    - Chemical Tagger
    - Gene Tagger 

<img src="https://user-images.githubusercontent.com/59030870/197599628-e47f1ec4-34a2-4aa0-ac7b-c5c11d9c5568.png" 
     width="530" 
     height="500"
   />


## Train the model

```sh
python trainBioCreative.py
```


## Requirements

  - python 3.7
  - torch 1.3
  - tqdm
  - transformers
  - numpy

## Citation
```sh
@inproceedings{esmail2022chemical,
  title={Chemical-Gene Relation Extraction with Graph Neural Networks and BERT Encoder},
  author={Esmail Zadeh Nojoo Kambar, Mina and Esmaeilzadeh, Armin and Taghva, Kazem},
  booktitle={The International Conference on Innovations in Computing Research},
  pages={166--179},
  year={2022},
  organization={Springer}
}
```
