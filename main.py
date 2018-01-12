#!/usr/bin/python3
import itertools
import multiprocessing
import pickle
from optparse import OptionParser
from os.path import dirname
from pprint import pprint

import numpy as np
import pandas as pd
from functools import reduce
import tqdm as tqdm
from sklearn import preprocessing

MAX_THREADS = multiprocessing.cpu_count() * 2 # count in hyperthreading
MAX_NUM_OF_RESULTS = 1000

stopword_set = None
options = None
idf_dict = {}
num_of_docs = 0


def define_cli_opts():
    USAGE = "%prog -q train-topics.list -d document.list -r train-run -o train-res.dat"
    result_opts = OptionParser(usage=USAGE)
    result_opts.add_option('-q', "--queries", dest='queries', help='file with a list of topic file names')
    result_opts.add_option('-d', "--documents", dest='documents', help='file with a list of document file names')
    result_opts.add_option('-r', "--label", dest='label', help='label identifying particular experiment run '
                                                               '(to be inserted in the result file as "run_id"')
    result_opts.add_option('-o', "--output-file", dest='output_file', help='output file  (Sec 5.5)')
    result_opts.add_option("--stopwords-removal", dest='stopwords', default="none",
                           help="method of removing stopwords. Choose from None (default), POS, frequency")
    result_opts.add_option("--lemmas", action='store_false', dest='forms', default=True,
                           help="use lemmas instead of forms")
    result_opts.add_option("--num-threads", dest='num_threads', default=MAX_THREADS, type="int",
                           help="number of threads for parallel computations")
    result_opts.add_option("--lowercase", action='store_true', dest='lowercase', default=False,
                           help="lowercase both document and query words")
    result_opts.add_option("--tf_weighting", dest='tf_weighting', type="string", default="natural",
                           help='how to weight term frequency in the document vector space. '
                                'Choose from "Natural" (default), "Log", "Boolean", "Augmented"')
    result_opts.add_option("--idf_weighting", dest='idf_weighting', type="string", default="none",
                           help='which inverse document frequency to use in the document vector space.'
                                'Choose from "None" (default), "Idf", "Probabilistic Idf"')
    result_opts.add_option("--similarity", dest='similarity', type="string", default="cosine",
                           help='which similarity measuring technique to use.'
                                'Choose from "Cosine" (default), "Dice"')
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
                if rank == MAX_NUM_OF_RESULTS:
                    break


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
    with open(query_file, encoding='utf-8') as input_file:
        for line in input_file.readlines():
            if is_tag_begin(line):
                cur_xml_tag = line[1:line.index('>')]
            elif line[0:2] == "</":  # tag ends
                pass
            elif cur_xml_tag == 'num':
                query_num = line
            elif cur_xml_tag == 'title':
                word = process_vert_format_line(line)
                if not word: continue
                query_string += " " + word

    return query_num.strip(), query_string.strip()


WORD_TYPE_INDEX = 3
NON_STOPWORD_TYPE_CHARS = ['A', 'C', 'N']


def is_stopword(result, line_split, removal_method):
    if removal_method.lower() == "none":
        return False
    elif removal_method.lower() == "pos":
        return line_split[WORD_TYPE_INDEX][0] not in NON_STOPWORD_TYPE_CHARS
    elif removal_method.lower() == "frequency":
        return result in stopword_set


def process_vert_format_line(line):
    result = ""
    word_index = 1 if options.forms else 2
    try:
        # get the form of the word
        line_split = line.split('\t')
        result = line_split[word_index]
        if is_stopword(result, line_split, options.stopwords):
            result = ""
        else:
            # get everything up to the first non-alphanum character
            result = "".join(itertools.takewhile(str.isalnum, result))
    except IndexError:
        pass
    return sanitize_word(result)


def convert_dict_to_dataframe(input_dict, cols, index):
    return convert_list_to_dataframe(list(zip(input_dict.keys(), input_dict.values())), cols, index)


def convert_list_to_dataframe(input_list, cols, index):
    return pd.DataFrame(input_list, columns=cols).set_index(index)


def is_tag_begin(line):
    return line[0] == '<' and line[1] != '/'


def sanitize_word(word):
    return word.lower() if options.lowercase else word


def count_words_in_doc(document_filename):
    cur_xml_tag = ""
    doc_num = ""
    wordcount = {}
    with open(document_filename, encoding='utf-8') as input_file:
        for line in input_file.readlines():
            if is_tag_begin(line):
                cur_xml_tag = line[1:line.index('>')].lower()
            elif line[0:2] == "</":  # tag ends
                pass
            if cur_xml_tag == 'docno':
                doc_num = line[line.find('>')+1:line.rfind('<')]
            elif cur_xml_tag in ['text', 'title', 'heading']:
                word = process_vert_format_line(line)
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


def get_max_and_sum_of_dict(wordcount):
    result_max = 0
    result_sum = 0
    for freq in wordcount.values():
        result_sum += freq
        if freq > result_max: result_max = freq
    return result_max, result_sum


def weigh_term_freq(wordcount):
    result = {}

    if options.tf_weighting.lower() in ["natural", "augmented"]:
        most_freqent_word_frequence, document_length = get_max_and_sum_of_dict(wordcount)

    for word, freq in wordcount.items():
        # tf part
        if options.tf_weighting.lower() == "boolean":
            word_weight = 1
        elif options.tf_weighting.lower() == "natural":
            word_weight = wordcount[word]/document_length
        elif options.tf_weighting.lower() == "log":
            word_weight = 1 + np.math.log10(wordcount[word])
        elif options.tf_weighting.lower() == "augmented":
            word_weight = 0.5 + 0.5*wordcount[word]/most_freqent_word_frequence
        else:
            raise ValueError("Unknown value of parameter --tf_weighting: " + options.tf_weighting.lower())

        # idf part
        try:
            idf_word_weight = idf_dict[word]
        except KeyError:
            idf_word_weight = 1

        if options.idf_weighting.lower() == "none":
            pass
        elif options.idf_weighting.lower() == "idf":
            word_weight = word_weight * np.math.log((float(num_of_docs)/float(idf_word_weight)))
        elif options.idf_weighting.lower() == "probabilistic idf":
            idf_part = max(0, np.math.log((float(num_of_docs - idf_word_weight) / float(idf_word_weight))))
            word_weight = word_weight * idf_part
        else:
            raise ValueError("Unknown value of parameter --idf_weighting: " + options.idf_weighting.lower())

        result[word] = word_weight

    return result


def process_document(filename):
    (doc_id, wordcount) = count_words_in_doc(filename)
    df_wordcount = convert_dict_to_dataframe(weigh_term_freq(wordcount), cols=["word", doc_id], index="word")
    return doc_id, wordcount, df_wordcount


def create_vector_space_from_docs(documents):
    docs_dir = dirname(documents)

    # count the number of documents
    global num_of_docs
    with open(documents, encoding='utf-8') as documents_file:
        for i, l in enumerate(documents_file, 1): pass
        num_of_docs = i

    with open(documents, encoding='utf-8') as documents_file:
        all_docs = list(map(lambda x: docs_dir + "/" + x.strip(), documents_file.readlines()))
        docs_info = map_parallel(process_document, all_docs)

    document_ids = [x[0] for x in docs_info]
    # word_count_dicts = [x[1] for x in docs_info]
    # collection_word_count_out = reduce(lambda x, y: join_dicts(x, y), word_count_dicts)
    # with open("obj/idf.pkl", "wb") as outp:
    #     pickle.dump(collection_word_count_out, outp)
    #     import sys
    #     sys.exit(0)

    # normalize vectors
    result_sparse_vector_space = [x[2] for x in docs_info]
    print("Normalizing vector space:")
    if options.num_threads == MAX_THREADS:
        document_vector_space = list(map_parallel(normalize, result_sparse_vector_space, threads=int(MAX_THREADS/2)))
    else:
        document_vector_space = list(map_parallel(normalize, result_sparse_vector_space, threads=options.num_threads))

    return document_vector_space, document_ids


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
    if options.similarity.lower() == "cosine":
        for query_word in df_query.index:
            if query_word in df_document.index:
                result += float(df_query.at[query_word, qry_id]) * float(df_document.at[query_word, doc_id])
    elif options.similarity.lower() == "dice":
        query = set(df_query.index)
        doc = set(df_document.index)
        intersect = query.intersection(doc)
        result = (float(2*len(intersect)))/(len(doc) + len(query))
    else:
        raise ValueError("Unknown value of parameter --similarity: " + options.tf_weighting.lower())

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
    scores, qry_id = query_scores
    global doc_list
    similar_docs = []
    for idx, score in enumerate(scores):
        if score > 0:
            similar_docs.append((doc_list[idx], score))
    return qry_id, sorted(similar_docs, key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    opts = define_cli_opts()
    (options, args) = opts.parse_args()

    # load precomputed idf for this document collection
    if options.idf_weighting in ["idf", "probabilistic idf"]:
        if options.lowercase:
            idf_filename = "obj/idf-lower.pkl"
        else:
            idf_filename = "obj/idf.pkl"
        with open(idf_filename, "rb") as inp:
            idf_dict = pickle.load(inp)

    if options.stopwords.lower() == "frequency":
        if options.lowercase:
            stopw_filename = "obj/stopwords-lower.pkl"
        else:
            stopw_filename = "obj/stopwords.pkl"
        with open(stopw_filename) as inp:
            stopword_set = pickle.load(inp)

    queries = parse_queries(options.queries)
    print("Queries parsed")

    print("Creating document vector space:")
    vector_space, doc_list = create_vector_space_from_docs(options.documents)
    print("Vector space created")

    # count words in queries
    query_ids = [query[0] for query in queries]
    queries = map(lambda x: (x[0], count_words_in_qry(x[1])), queries)

    # transform queries into dataframes
    queries = map(lambda x: (x[0], convert_dict_to_dataframe(x[1], cols=["word", x[0]], index="word")), queries)

    # normalize
    normalized_queries = list(map(lambda x: normalize(x[1]), queries))

    print("Computing query - document similarities and ranking documents...")
    # for each query count the similarity between it and all the documents
    similarity = map(compute_all_similarities, normalized_queries)

    # rank documents based on said similarity
    results = map(get_relevant_docs_for_qry, list(zip(similarity, query_ids)))
    print("Ranking done")

    pprint('Writting results...')
    write_output_file(options.output_file, list(results), options.label)
    pprint('Writting results done')
