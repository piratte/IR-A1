#!/usr/bin/python3

import pickle
import sys

with open(sys.argv[1] + ".pkl", "rb") as inp:
    input_dict = pickle.load(inp)

result_dict = {}
for key, val in input_dict.items():
    new_key = key.lower()
    if key not in result_dict:
        result_dict[new_key] = val
    else:
        result_dict[new_key] += val

with open(sys.argv[1] + "-lower.pkl", "wb") as outp:
    pickle.dump(result_dict, outp)
