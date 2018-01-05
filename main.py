#!/usr/bin/python3
from optparse import OptionParser
from pprint import pprint
from os.path import dirname

import itertools
import pandas as pd
import numpy as np
from functools import reduce
import tqdm as tqdm
from sklearn import preprocessing

import multiprocessing

MAX_THREADS = multiprocessing.cpu_count() * 2 # count in hyperthreading


def define_cli_opts():
    USAGE = "%prog -q train-topics.list -d document.list -r train-run -o train-res.dat"
    result_opts = OptionParser(usage=USAGE)
    result_opts.add_option('-q', "--queries", dest='queries', help='file with a list of topic file names')
    result_opts.add_option('-d', "--documents", dest='documents', help='file with a list of document file names')
    result_opts.add_option('-r', "--label", dest='label', help='label identifying particular experiment run '
                                                               '(to be inserted in the result file as "run_id"')
    result_opts.add_option('-o', "--output_file", dest='output_file', help='output file  (Sec 5.5)')

    return result_opts


def write_output_file(output_file, results, label):
    """
    Wanted output columns are:
    1. qid
    2. iter
    3. docno
    4. rank
    5. sim
    6. run_id
    :param output_file: output file path
    :param results: list of results [(qid, [(doc_no, sim),...]),...]
    :param label: run_id
    """
    with open(output_file, mode='w') as outfile:
        for qry in results:
            qid = qry[0]
            rank = 0
            for doc_res in qry[1]:
                outfile.write("%s\t0\t%s\t%d\t%f\t%s\n" % (qid, doc_res[0], rank, doc_res[1], label))
                rank += 1


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


def convert_dict_to_dataframe(input_dict, cols, index):
    return convert_list_to_dataframe(list(zip(input_dict.keys(), input_dict.values())), cols, index)


def convert_list_to_dataframe(input_list, cols, index):
    return pd.DataFrame(input_list, columns=cols).set_index(index)


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


def join_dicts(dict_old, dict_new):
    intersection = {key: val + dict_old[key] for key, val in dict_new.items() if key in dict_old}
    result = dict_old.copy()
    result.update(dict_new)
    result.update(intersection)
    return result


def map_parallel(funct_to_map, input_list, threads=MAX_THREADS):
    with multiprocessing.Pool(threads) as pool:
        results = list(tqdm.tqdm(pool.imap(funct_to_map, input_list), total=len(input_list)))

    return results


def process_document(filename):
    (doc_id, wordcount) = count_words_in_doc(filename)
    df_wordcount = convert_dict_to_dataframe(wordcount, cols=["word", doc_id], index="word")
    return doc_id, wordcount, df_wordcount


def create_vector_space_from_docs(documents):
    docs_dir = dirname(documents)
    with open(documents) as documents_file:
        all_docs = list(map(lambda x: docs_dir + "/" + x.strip(), documents_file.readlines()))
        docs_info = map_parallel(process_document, all_docs)

    # TODO: calculate tf-idf -> create a vector space form collection

    # normalize vectors
    document_ids = [x[0] for x in docs_info]
    word_count_dicts = [x[1] for x in docs_info]
    # collection_word_count_out = reduce(lambda x, y: join_dicts(x, y), word_count_dicts)
    result_sparse_vector_space = [x[2] for x in docs_info]
    print("Normalizing vector space:")
    document_vector_space = list(map_parallel(normalize, result_sparse_vector_space))
    return document_vector_space, {}, document_ids


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
    Normalize dataframe and return it as a dataframe with same properties
    :param dataframe: input dataframe
    :return: dataframe with normalized values
    """
    cols = [dataframe.index.name]
    cols.extend(dataframe.columns.values)
    return convert_list_to_dataframe(list(zip(dataframe.index, preprocessing.normalize(dataframe, norm='l2', axis=0))),
                                     cols=cols, index=dataframe.index.name)


def compute_similarity(df_document, df_query):
    result = 0
    doc_id = df_document.columns[0]
    qry_id = df_query.columns[0]
    for query_word in df_query.index:
        if query_word in df_document.index:
            result += float(df_query.at[query_word, qry_id]) * float(df_document.at[query_word, doc_id])

    return result


def compute_all_similarities(df_query):
    """
    Computes similarity between query and documents in vector space (uses global variable to access the VS)
    :param df_query: query represented as a dataframe (index=word, 1 column=weight)
    :return: similarity of the documents represented in teh vector_space to the query
    """
    global vector_space
    return list(map(lambda df_document: compute_similarity(df_document, df_query), vector_space))


def get_relevant_docs_for_qry(query_scores):
    scores, qry_id= query_scores
    global doc_list
    similar_docs = []
    for idx, score in enumerate(scores):
        if score > 0:
            similar_docs.append((doc_list[idx], score))
    return qry_id, sorted(similar_docs, key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    opts = define_cli_opts()
    (options, args) = opts.parse_args()
    queries = parse_queries(options.queries)
    print("Queries parsed")

    print("Creating document vector space:")
    vector_space, collection_word_count, doc_list = create_vector_space_from_docs(options.documents)
    print("Vector space created")

    # count words in queries
    query_ids = [query[0] for query in queries]
    queries = map(lambda x: (x[0], count_words_in_qry(x[1])), queries)

    # transform queries into dataframes
    queries = map(lambda x: (x[0], convert_dict_to_dataframe(x[1], cols=["word", x[0]], index="word")), queries)

    # normalize
    normalized_queries = list(map(lambda x: normalize(x[1]), queries))

    print("Computing query - document similarities:")
    # for each query count the similarity between it and all the documents
    # similarity = map(compute_all_similarities, list(normalized_queries))
    similarity = map_parallel(compute_all_similarities, normalized_queries)
    print("Similarities computed")

    # rank documents based on said similarity
    print("Ranking documents:")
    results = map_parallel(get_relevant_docs_for_qry, list(zip(similarity, query_ids)))
    print("Ranking done")

    pprint("Writting results...")
    write_output_file(options.output_file, list(results), options.label)
    pprint("Writting results done")
