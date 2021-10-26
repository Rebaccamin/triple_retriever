# Triple_retriever

This repository contains the codes for our paper "Triple-Fact Retriever: An explainable reasoning retrieval model for multi-hop QA problem.".

## Pre-process. 
  
  We apply two existing OpenIE tools, [MinIE](https://github.com/uma-pi1/minie#minie-open-information-extraction-system) and [OpenIE]( https://stanfordnlp.github.io/CoreNLP/openie.html) to do the triple extraction over the text corpus. We first conduct [correference resolution](https://github.com/huggingface/neuralcoref) over the raw documents and then split the documents by NLTK sentence tokenizer into a set of sentences.
  
  For the hotpotQA data, as each documement instance is consisted of a title and its corresponding text，e.g, 
  >{"Barack Obama":"Barack Hussein Obama II is an American politician, author, and retired attorney who served as the 44th president of the United States from 2009 to 2017. A member of the Democratic Party, Obama was the first African-American president of the United States. He previously served as a U.S. senator from Illinois from 2005 to 2008 and as an Illinois state senator from 1997 to 2004. Outside of politics, Obama has published three bestselling books; Dreams from My Father (1995), The Audacity of Hope (2006) and A Promised Land (2020)"}.
  
  We deploy a local correference resolution method by the title. Details in **extract_triples_minie.py and extract_triples_stanford.py** .

## Triple extractor

   Merge_triples.py: we use the union of two extracted triple set after Pre-process without depulicates.

   triple_cluster.py: this code is the implementation of our Algorithm 1 in the paper.
   
   For the evaluation, we dump the documents and the corresponding triple sets to [Elastic search](https://www.elastic.co/guide/en/elasticsearch/reference/7.15/install-elasticsearch.html), we apply the conventional token-based method to do the query-document search. 
   Details in **es_build.py and evaluation.py**.

## Retriever


## Dataset:
1. [HotpotQA](https://hotpotqa.github.io/).
2. [Wikihop](https://qangaroo.cs.ucl.ac.uk/).
   As this dataset is only contain the input NL query and grounded answer to the query. We generate the grounded documents by ourselves. According to its paper, they provided a set of support documents, which contain the exact clue to fetch the answer. We generate the hop 1 grounded document by the subject entity in the question.
