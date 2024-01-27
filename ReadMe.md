# Topic Classification of Publications
## Table of Contents
* [Overview](#overview)
* [Getting started](#getting-started)
  * [Getting the data](#getting-the-data)
  * [Using OpenAI Embeddings](#using-openai-embeddings)
  * [What does each file do?](#what-does-each-file-do)
* [How to run the project](#how-to-run-the-project)
* [Acknowledgement](#acknowledgement)
## Overview
This project is done as part of Technical University of Delft's [Research Project 2023-24](https://github.com/TU-Delft-CSE/Research-Project).
The project explores effective ways of topic classification for publication in scientific field. It mainly focuses on utilizing OpenAI Embeddings (text-embedding-ada-002 model) and XGBoost combination on 'April 2022 Crossref Data'. Alexandria3k (A3k) is used for populating/extracting data efficiently as the data set is large (168GB). 

## Getting started

### Getting the data
Download Crossref data set by following the guideline provided by [A3k](https://dspinellis.github.io/alexandria3k/index.html)
```
aria2c https://doi.org/10.13003/83b2gq
```
### Using OpenAI Embeddings
Ensure that you have created an API key created. Keep in mind that using the API for this project requires credit on your account as text-embedding-ada-002 model is not a free service. Follow the guidelines provided by [OpenAI](https://platform.openai.com/docs/api-reference/introduction) to set API correctly.

Also, you can consult documentation about [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings/use-cases
) for more details.

### What does each file do?
'initial-classifications' folder contains files that have been created for exploring different combinations of fields from the data set for classification task: abstract only, abstract + title, abstract + title + author.

Under 'initial-classifications' file, you see another folder named 'bm25'. This contains files for comparing the results of BM25 + XGBoost and OpenAI Embeddings + XGBoost using abstract. 

As it can be seen in the name of the .py files, there are numbers. These numbers indicate the order of how the files should be ran. As each file with same number in different folders does same, 'final-classification' folder is used as an example of what each file does.

1. 1-get-works.py gets all the works that have abstract, title, work_names (also known as topics). This file is also in 'initial-classifications' folder. 1-stratify-works is specific for 'final-classification' folder as this is where the stratification process is introduced. 
2. 2-clean-abstract.py cleans abstracts to ensure higher quality of results and is also present in 'initial-classifications' folder. 2.1-token-estimation gives an estimation of how many tokens there will be for the given input. This is useful in estimating the cost for OpenAI Embeddings. This can be copied and used for other folders before running the next steps. 
3. 3-openai-xgboost-title.py contains code for running the OpenAI Embeddings and XGBoost. This is where the actual topic classification is done. This can be also found in 'initial-classifications' folder. 
4. 4-grid-search-stratified-samples.py has code for tuning the hyper-parameters using GridSearchCV. This is also present in 'initial-classifications' folder. 

## How to run the project
The way to run files under folder 'initial-classifications', 'initial-classifications/bm25' and 'final-classification' is identical. Following steps uses 'final-classification' folder as an example.

1. Go to the folder location where the file you want to run is located.
```
cd final-classification
```
2. Assuming that you have the data set ready, run the files in order. For step 1, you have two options so please read carefully and follow the instructions accordingly (Be careful since specific names of files and tables differ for code in 'initial-classifications' folder).

You can populate and save the data set as .csv file using A3k with SQL queries in files that start with 1. Then, convert .csv file to 'works-with-abstract-title-topics.db' file with 'expanded_abstract' as the table name. 

```
a3k query crossref 'April 2022 Public Data File from Crossref' \
  --query 'add SQL query' \
  --output crossref_data.csv \
  --output-encoding use utf-8-sig
```
or

Run the .py file in the case of memory shortage while converting .csv to .db. This requires 'April 2022 Public Data File from Crossref' to be saved in .db file. 

```
python 1-get-works.py
```

The rest of the steps can be ran by using following command format:
```
python 1.1-stratify-works.py
```

_Note: In step 3, for the first time running do not modify anything and run the code. This runs OpenAI Embeddings and the embeddings will be saved in a .csv file. For second or more runs, please comment out the OpenAI Embedding part (e.g. line 12-33 in 3-openai-xgboost-title.py) and uncomment the lines that read the saved embeddings (e.g. line 35-37 in 3-opani-xgboost-title.py) in order to avoid unnecessary cost._

_Intermediate output files, like .csv and .db files, are included in the repository due to size limitation._

## Acknowledgement
I would like to give special thanks to Professor Diomidis Spinellis and Professor Georgios Gousios for the guidance throughout the project.