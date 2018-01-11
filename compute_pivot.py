import pickle
from pprint import pprint

import numpy as np
import sys

NUM_OF_BINS = 100
QUERY_NUM = 25

def get_relevancy_to_dict(doc_list, relevacy_dict):
    result = []
    for doc_bin in bin_doc_list:
        bin_rel = 0
        for doc in doc_bin:
            if doc in doc_real_relevancy:
                bin_rel += doc_real_relevancy[doc]
        bin_rel /= float(len(doc_bin))
        result.append(bin_rel)
    return result


with open(sys.argv[1], "rb") as inp:
    doc_metric = pickle.load(inp)

doc_metric_sorted = sorted(list(doc_metric), key=lambda x: x[1])

# cut off the outliers on the tail
doc_metric_sorted_interesting = doc_metric_sorted[:80000]

hist = np.histogram([x[1] for x in doc_metric_sorted_interesting], bins=NUM_OF_BINS)
hist_counts = hist[0]
hist_bin_limits = hist[1]

bin_doc_list = [[] for i in range(0, NUM_OF_BINS)]
cur_bin = 0
for doc in doc_metric_sorted_interesting:
    if doc[1] > hist_bin_limits[cur_bin+1]:
        cur_bin += 1
    bin_doc_list[cur_bin].append(doc[0])

doc_real_relevancy = {}
with open("qrels-train.txt") as inqrels:
    for l in inqrels:
        l = l.split()
        if l[2] not in doc_real_relevancy:
            doc_real_relevancy[l[2]] = int(l[3])
        else:
            doc_real_relevancy[l[2]] += int(l[3])

real_bin_rels = get_relevancy_to_dict(bin_doc_list, doc_real_relevancy)

doc_computed_relevancy = {}
with open(sys.argv[2]) as inqrels:
    for l in inqrels:
        l = l.split()
        if l[2] not in doc_real_relevancy:
            doc_real_relevancy[l[2]] = 1
        else:
            doc_real_relevancy[l[2]] += 1

computed_bin_rels = get_relevancy_to_dict(bin_doc_list, doc_computed_relevancy)

real_vs_comp_rel = map(lambda x: x[0] > x[1], zip(real_bin_rels, computed_bin_rels))
pprint(list(real_vs_comp_rel).index(True))
