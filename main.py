#!/usr/bin/python3
from optparse import OptionParser
from pprint import pprint
from os.path import dirname

import itertools
import pandas as pd


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


def is_tag_begin(line):
    return line[0] == '<' and line[1] != '/'


def sanitize_doc_word(word):
    return word


def count_words(document_filename):
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
            (doc_id, wordcount) = count_words(docs_dir + "/" + document.strip())
            df_wordcount = pd.DataFrame(list(zip(wordcount.keys(), wordcount.values())), columns=["words", "count"])\
                .set_index("words")
            document_word_count.append(df_wordcount)
            document_ids.append(doc_id)

        document_word_count = pd.concat(document_word_count, axis=1)
        document_word_count.columns = document_ids
        document_word_count = document_word_count.fillna(0)

    # TODO: sum word occurrences in collection

    # TODO: calculate tf-idf -> create a vector space form collection


if __name__ == "__main__":
    opts = define_cli_opts()
    (options, args) = opts.parse_args()
    queries = parse_queries(options.queries)

    vector_space = create_vector_space_from_docs(options.documents)

