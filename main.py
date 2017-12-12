#!/usr/bin/python3
from optparse import OptionParser
from pprint import pprint
from os.path import dirname

import itertools
import pandas as pd
import numpy as np
from sklearn import preprocessing


def define_cli_opts():
    USAGE = "%prog -q train-topics.list -d document.list -r train-run -o train-res.dat"
    result_opts = OptionParser(usage=USAGE)
    result_opts.add_option('-q', "--queries", dest='queries', help='file with a list of topic file names')
    result_opts.add_option('-d', "--documents", dest='documents', help='file with a list of document file names')
    result_opts.add_option('-r', "--label", dest='label', help='label identifying particular experiment run '
                                                               '(to be inserted in the result file as "run_id"')
    result_opts.add_option('-o', "--output_file", dest='output_file', help='output file  (Sec 5.5)')

    return result_opts


def parse_queries(queries_filename):
    result = []
    data_dir = dirname(queries_filename)
    with open(queries_filename) as input_files_file:
        for query_file in input_files_file.readlines():
            result.append(process_query_file(data_dir + "/" + query_file.strip()))
    return result


def process_query_file(query_file):
    cur_xml_tag = ""
    query_num = ""
    query_string = ""
    # TODO: check if this is UTF-8
    with open(query_file) as input_file:
        for line in input_file.readlines():
            if is_tag_begin(line):
                cur_xml_tag = line[1:line.index('>')]
            elif line[0:2] == "</":  # tag ends
                pass
            elif cur_xml_tag == 'num':
                query_num = line
            elif cur_xml_tag == 'title':
                query_string += " " + process_vert_format_line(line)

    return query_num.strip(), query_string.strip()


def process_vert_format_line(line):
    result = ""
    try:
        result = line.split('\t')[2] # get the form of the word

        # get everything up to the first non-alphanum character
        result = "".join(itertools.takewhile(str.isalnum, result))
    except IndexError:
        pass
    return result


def convert_to_dataframe(input_dict, cols, index):
    return pd.DataFrame(list(zip(input_dict.keys(), input_dict.values())), columns=cols).set_index(index)

def is_tag_begin(line):
    return line[0] == '<' and line[1] != '/'


def sanitize_doc_word(word):
    return word


def count_words_in_doc(document_filename):
    cur_xml_tag = ""
    doc_num = ""
    wordcount = {}
    with open(document_filename) as input_file:
        for line in input_file.readlines():
            if is_tag_begin(line):
                cur_xml_tag = line[1:line.index('>')].lower()
            elif line[0:2] == "</":  # tag ends
                pass
            if cur_xml_tag == 'docno':
                doc_num = line[line.find('>')+1:line.rfind('<')]
            elif cur_xml_tag == 'text':
                word = sanitize_doc_word(process_vert_format_line(line))
                if not word: continue
                if word not in wordcount:
                    wordcount[word] = 1
                else:
                    wordcount[word] += 1

    return doc_num, wordcount


def create_vector_space_from_docs(documents):
    document_ids = []
    document_word_count = []
    docs_dir = dirname(documents)
    with open(documents) as documents_file:
        for document in documents_file.readlines():
            (doc_id, wordcount) = count_words_in_doc(docs_dir + "/" + document.strip())
            df_wordcount = convert_to_dataframe(wordcount, cols=["word", doc_id], index="word")
            document_word_count.append(df_wordcount)

    document_word_count = pd.concat(document_word_count, axis=1)
    document_word_count = document_word_count.fillna(0)

    # sum word occurrences in collection
    collection_word_count_out = document_word_count.sum(axis=1)

    # TODO: calculate tf-idf -> create a vector space form collection

    # normalize vectors
    document_vector_space = normalize(document_word_count)

    return document_vector_space, collection_word_count_out, document_ids


def count_words_in_qry(query):
    result = {}
    for word in query.split():
        if word not in result:
            result[word] = 1
        else:
            result[word] += 1
    return result


def normalize(dataframe):
    """
    For the input pandas dataframe output a normalized numpy array
    """

    return preprocessing.normalize(dataframe, norm='l2', axis=0)


if __name__ == "__main__":
    opts = define_cli_opts()
    (options, args) = opts.parse_args()
    queries = parse_queries(options.queries)
    print("Queries parsed")

    vector_space, collection_word_count, doc_list = create_vector_space_from_docs(options.documents)
    print("Vector space created")

    # count words in queries
    queries = map(lambda x: (x[0], count_words_in_qry(x[1])), queries)

    # transform queries into dataframes
    queries = map(lambda x: (x[0], convert_to_dataframe(x[1], cols=["word", x[0]], index="word")), queries)

    # transform them into a vector the same size as the document vector space
    query_word_counts = [x[1] for x in queries]
    query_word_counts.append(collection_word_count)
    df_query_word_count = pd.concat(query_word_counts, axis=1)

    # drop the collection_word_count column and the rows for words which are in queries, but not in documents
    df_query_word_count = df_query_word_count.drop(df_query_word_count.columns[-1], axis=1)\
                                             .fillna(0)\
                                             [df_query_word_count.index.isin(collection_word_count.index)]

    # normalize
    normalized_queries = normalize(df_query_word_count)

    # for each query count the similarity between it and all the documents
    similarity = np.dot(np.transpose(normalized_queries), vector_space)
    print("Similarity computed")
    pprint(similarity.shape)

    # rank documents based on said similarity
