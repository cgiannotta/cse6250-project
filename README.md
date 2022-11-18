# CSE6250: Big Data for Health Informatics Course Project

## Description

This repository holds code for the final course project for Georgia Tech’s CSE6250 course (Big Data for Health Informatics). The project is meant to replicate the analysis from the paper Comparing Deep Learning and Concept Extraction Based Methods for Patient Phenotyping from Clinical Narratives, by Gehrmann at al.

## How to install and run:
1. Download MIMIC-III data and join with annotations from authors. Alternatively, you can download their word2vec embeddings here.
2. Download our annotations.csv file (add link) and run python preprocess.py data/annotations.csv w2v.txt. This outputs two h5 files, one with data split into batches and one without batches. 

### Basic Models
3. Run python basic_models.py --data data-nobatch.h5 --ngram 5 to get baseline performance. You can uncomment the classifier lines (74-76) in basic_models.py to specify which type of basic model (Logistic, Naive Bayes, or SVM) you wish to run.

### CNN Model
4. Open up the CNN.ipynb notebook in Google Colab to run the CNN experiments, it may be necessary to update the system path locations in the Setup section of the notebook. 
5. A Wandb account added to the project team is required to run the wandb code as-is
6. In the Experiment section of the notebook, the sweeps config can be modified to change parameter values. In Wandb, a sweep allows you to run multiple experiments with a randomized set of configurations from a pre-defined sweep configuration. 

## How to use

This methodology could be used to analyze a variety of text data, though the more similar it is to the MIMIC discharge summaries the more likely the model is to give similar output.

## Credits

References and credits for the work in this repository can be found in our final report here (add link).
