# Information Retrieval Assignment 1

## Goal

The goal is to get familiar with vector space models in Information Retrieval, 
text pre-processing, system tuning, and experimentation.

## Technical documentation

The solution is using the python 3 programming language. Non standard libraries 
include pandas, numpy, sklearn and tqdm, which can all be easily installed using
pythons extending tools (e.g. pip). 

To speed up the computing time, the program runs in multiple processes. For thread
management, the pythons native multiprocessing library is used. The program 
detects the number of available processors and runs twice as many processes, because
the parallelized tasks mainly involve loading data. 

Because this program is aimed to be used with a set collection of documents, 
there are several precomputed data structures that may be loaded during runtime for
performance reasons. The loading of these is handled by the pythons native pickle
library. 

The expected runtime is 5-10 minutes, but it may wary due on the combination 
of parameters and performance of the computer it is being run on.

## Build 

Since python does not need any compilation, there is no build required. However
to make things easier a makefile is included, which handles downloading the needed
precomputed results. Run ```make``` in the directory with the Makefile.

## Usage

The program is meant to be ran in a command line and has been tested only on
Linux in bash (see the full list of options bellow), which is sufficient according 
to the assignment specifications. Using other operating systems or setups 
although in general possible is not recommended. 

It is recomended to state the whole path to the input files (the topics and document lists).
The documents are expected to be in a documents folder in the same location as the
input files.`` 


```
main.py -q train-topics.list -d document.list -r train-run -o train-res.dat [other_options]

Options:
  -h, --help            show this help message and exit
  -q QUERIES, --queries=QUERIES
                        file with a list of topic file names
  -d DOCUMENTS, --documents=DOCUMENTS
                        file with a list of document file names
  -r LABEL, --label=LABEL
                        label identifying particular experiment run (to be
                        inserted in the result file as "run_id"
  -o OUTPUT_FILE, --output-file=OUTPUT_FILE
                        output file  (Sec 5.5)
  --num-threads=NUM_THREADS
                        number of threads for parallel computations
  --lemmas              use lemmas instead of forms
  --lowercase           lowercase both document and query words
  --stopwords-removal=STOPWORDS
                        method of removing stopwords. Choose from None
                        (default), POS, frequency
  --tf_weighting=TF_WEIGHTING
                        how to weight term frequency in the document vector
                        space. Choose from "Natural" (default), "Log",
                        "Boolean", "Augmented"
  --idf_weighting=IDF_WEIGHTING
                        which inverse document frequency to use in the
                        document vector space.Choose from "None" (default),
                        "Idf", "Probabilistic-Idf"
  --similarity=SIMILARITY
                        which similarity measuring technique to use.Choose
                        from "Cosine" (default), "Dice"
  --query_norm=QUERY_NORM
                        which normalization technique to use on the
                        queries.Choose from "Cosine" (default), "None",
                        "Unique
  --doc_norm=DOC_NORM   which normalization technique to use on the
                        documents.Choose from "Cosine" (default), "None",
                        "Unique
```

## Experiments

As you can see from the number of options, the program is designed to be as 
modular as possible. After some initial debugging the program was ran with 
all meaningful combinations of parameters. The best combination for the training
set of queries the results of which are submitted as run-1 is:

| option                 | value           |
|------------------------|-----------------|
| term extraction        | lemmas          |
| stopword removal       | frequency based |
| tf-weighting           | natural         |
| idf-weighting          | idf             |
| similarity             | cosine          |
| query normalization    | none            |
| document normalization | cosine          |

Results can be replicated running command:
```./run -q /home/madam/IR/A1/train-topics.list -d /home/madam/IR/A1/documents.list -r run-1 -o run-1_res.dat --lemmas  --stopwords-removal frequency --tf_weighting natural --idf_weighting idf --similarity cosine --query_norm none --doc_norm cosine```

## Implementation notes

Frequency based stopword removal removes the most frequent words (500 when the 
terms are not lowercased, 73 when they are). These numbers were picked by 
examining the most frequent words and picking a line, above which there were no
words that might be of any importance. A more experimental method might yielded
better results.

Pivot normalization was not implemented. The reason was, that we were unable to
find the pivot: this method returns a lot of potential results although with small
similarity; there is a lot less actually relevant documents. The cosine 
similarity is not disadvantaging the short documents enough for us to see the 
intersection of the probabilities (see the lecture handouts).