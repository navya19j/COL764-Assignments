# COL 764 Assignment 1 : Inverted Index Construction

## Description

The aim of Assignment 1 is to build an ecient Boolean retrieval system for English corpora using ecient
inverted index structures. 

## Pre-Steps

- pip install the following : python-snappy, BeautifulSoup

## Running the code

The submission contains two bash shell scripts and three python files.

1. invidx_cons.py : Reads the input directory, preprocesses the data and generates the dictionary and postings list. It also applies compression schemes to the postings list.  If the name of the postings list file is specified as indexfile, the program will generate two output files.
    1. indexfile.dict : The inverted token → start and length of the postings list of the token in the common postings list, for the training data. It also stores the compression scheme used, the number of zeros added to write the file into bytes and the original docIDs to assigned integer docIDs.
    2. indexfile.idx : The common postings list of all the tokens in the vocabulary.

    Use command bash [invidx.sh] coll-path indexfile stopwordfile 0|1|2|3|4|5 xml-tags-info to run the code.

    coll-path specifies the directory containing the files containing documents
    of the collection

    indexfile is the name of the index files that will be generated by the
    program.

    stopwordfile is a file that contains the stopwords that should be eliminated.

    0|1|2|3|4|5 these specify the compressions that will be applied – 0 specifies
    no compression, and 1—5 correspond to c1 to c5 as
    listed above. 

    xml-tags-info is a file that contains the tag for document identifier in the
    first line, followed by tags that contain the indexable portion
    in each line, starting from the second line.

2. [boolsearch.py] : Reads the query file and retrieves all the documents containing all the terms of the query except stopwords.

    Use command bash boolsearch[.sh] queryfile resultfile indexfile dictfile

    queryfile a file containing keyword querie
    resultfile the output file

    indexfile the index file generated by invidx cons program
    dictfile the dictionary file generated by the invidx cons program