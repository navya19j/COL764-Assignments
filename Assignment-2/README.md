# COL 764 Assignment 2 : Document Reranking Task

## Description

The aim of the assignment is to develop “telescoping” models aimed at improving the precision of results using pseudo-relevance feedback.

## Pre-Steps

- pip install the following : BeautifulSoup, nltk, numpy

## Topics Covered

- Pre-Processing Dataset

- Pseudo-relevance Feedback with Rocchio’s method

- Hyperparameter Tuning of a,b,y using grid search

- Relevance Model based Language Modeling

- Hyperparameter Tuning of u in Dirchlet Smoothing

## Running the code

The submission contains two bash shell scripts and two python files.

1. utils.py : Reads the input directory, preprocesses the data and generates the doc_freq dictionary. This Python file applies the rocchio algorithm to re-rank the top-100 retrieved documents. If the name of the output file is specified as out.txt, the program will generate one output file :
    1. out.txt : Contains the re-ranking of the top-100 documents for each query in the trec_eval format

    Use command bash [rocchio rerank.sh] [query-file] [top-100-file] [collection-file] [outputfile]

    query-file: file containing the queries in the same xml form as the training queries released

    top-100-file: a file containing the top100 documents in the same format as train and dev top100 files given, which need to be reranked

    collection-dir: directory containing the full document collection. Specifically, it will have metadata.csv, a subdirectory named document parses which in turn contains subdirectories pdf json and pmc json.

    output-file: file to which the output in the trec eval format has to be written 

2. lm.py : Reads the query file and retrieves all the documents containing all the terms of the query except stopwords. This Python file applies the Lavrenko and Croft’s relevance models with an Unigram model with Dirichlet smoothing algorithm to re-rank the top-100 retrieved documents. If the name of the output file is specified as out.txt and the expansions file as expand.txt, the program will generate two output file :
    1. out.txt : Contains the re-ranking of the top-100 documents for each query in the trec_eval format

    2. expand.txt : contains the expansion terms for each query in the format :
    <topic number> : <expansion term1>, ... <expansion term20>

    Use command bash lm rerank.sh [rm1|rm2] [query-file] [top-100-file] [collection-dir]
[output-file] [expansions-file]
    
    query-file: file containing the queries in the same xml form as the training
    queries released
    
    top-100-file: a file containing the top100 documents in the same format as
    train and dev top100 files given, which need to be reranked
    collection-dir: directory containing the full document collection. Specifically,
    it will have metadata.csv, a subdirectory named document
    parses which in turn contains subdirectories pdf json
    and pmc json.
    
    output-file: file to which the output in the trec eval format has to be written
    
    rm1|rm2: (only for LM) specifies if we are using RM1 or RM2 variant of
    relevance model.
    
    expansions-file: (only for LM) specifis the file to which the expansion terms used
    for each query should be output.

3. Trec_Eval : Use command with output re-ranked documents to get the NDCG scores using the TREC tool
./trec_eval -m ndcg qrels.txt output.txt

where qrels.txt contains the relevance judgements