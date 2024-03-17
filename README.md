# Emergent Positional Embeddings

*Final Project for LING 334: Computational Linguistics*

This project seeks to extend results from [Neel Nanda's quick investigation](https://www.lesswrong.com/posts/Ln7D2aYgmPgjhpEeA/tiny-mech-interp-projects-emergent-positional-embeddings-of) of whether language models like GPT-2 learn emergent representations of relative word position, a.k.a. "is this token the 2nd word in the sentence?"

## Paper

[Probing for Emergent Relative Word Representation in GPT-2](https://static.us.edusercontent.com/files/J14kUvoC0ZBZrpKohTCuJrON)

## Abstract

> This paper investigates emergent relative word representation in GPT-2, motivated by the challenges posed by token representations in transformers. Inspired by Neel Nanda's informal exploration and the general project of mechanistic interpretability, this project extends beyond toy inputs to deal with the challenge of probing on natural, cohesive sentences. We provide visualizations and detailed training statistics on linear probing to illustrate the abstract representations of information in GPT-2.

## Usage

```
conda env create -f environment.yml
conda activate pos-embed
python -m spacy download en_core_web_sm
```

See `probe_model.py` for definition of linear probe class and training loop.

See `train_probes.ipynb` for walkthrough of generating "Random Words" dataset and training probes.

See `train_probes_sentence.ipynb` for walkthrough of generating embeddings from natural sentences and training probes.

See `visuals.ipynb` for code to create all core visuals of the report.
