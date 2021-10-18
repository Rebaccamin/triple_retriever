import os
import json
from nltk.tokenize import sent_tokenize


def triple_extraction_minie(sentence,port=8080):
    shell = '''curl \'http://localhost:8080/minie/query\' -X POST -d \''''+sentence+'''\' '''
    result_str = os.popen(shell).read()

    facts=json.loads(result_str)
    triples=[]
    for t in facts['facts']:
        triples.append([t['subject'],t['predicate'],t['object']])
    return triples


def split2sen(text):
    return sent_tokenize(text)


