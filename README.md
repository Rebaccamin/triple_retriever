# Triple_retriever

This repository contains the codes for our paper "Triple-Fact Retriever: An explainable reasoning retrieval model for multi-hop QA problem".

## Pre-process. 
  
  We apply two existing OpenIE tools, [MinIE](https://github.com/uma-pi1/minie#minie-open-information-extraction-system) and [OpenIE]( https://stanfordnlp.github.io/CoreNLP/openie.html) to do the triple extraction over the text corpus. We first conduct [correference resolution](https://github.com/huggingface/neuralcoref) over the raw documents and then split the documents by NLTK sentence tokenizer into a set of sentences.
  
  For the hotpotQA data, as each documement instance is consisted of a title and its corresponding text，e.g, 
  >{"Barack Obama":"Barack Hussein Obama II is an American politician, author, and retired attorney who served as the 44th president of the United States from 2009 to 2017. A member of the Democratic Party, Obama was the first African-American president of the United States. He previously served as a U.S. senator from Illinois from 2005 to 2008 and as an Illinois state senator from 1997 to 2004. Outside of politics, Obama has published three bestselling books; Dreams from My Father (1995), The Audacity of Hope (2006) and A Promised Land (2020)"}.
  
  We deploy a local correference resolution method by the title. Details in **extract_triples_minie.py and extract_triples_stanford.py** .

## Triple extractor

   Merge_triples.py: we use the union of two extracted triple set after Pre-process without depulicates.

   triple_cluster.py: this code is the implementation of Algorithm 1 in the paper.
   
   For the evaluation, we dump the documents and the corresponding triple sets to [Elastic search](https://www.elastic.co/guide/en/elasticsearch/reference/7.15/install-elasticsearch.html), we apply the conventional token-based method to do the query-document search. 
   Details in **es_build.py and evaluation.py**.

## Retriever
   
   The retriever for the multi-hop question documents is an iteration work, for each hop, we deploy the single hop retriever to retrieve the topk documents. Given a NL question, a set of candidate documents (which is composed of the triple facts), **train_one_hop.py** is used to find the ground document by its maximum matched triple fact to the input query.
   
   Train the model by:
   
   `bash run.sh`
   
   After each hop, we use **query_generator.py** to fuse the retrieved triple knowledge into the last-hop query to generate a new query for next hop retriever.
    
   Specifically, as the original dataset lacks of grounded supervison of intermediate query for the query generator training, we simulate [GoldEn](https://github.com/qipeng/golden-retriever) to generate the ground intermedidate query. 

## Dataset:
1. [HotpotQA](https://hotpotqa.github.io/). It is a human-annotated large-scale multi-hop QA dataset.There are 90447 questions for training and 7405 questions for testing. We conduct our experiments on the *full wiki* setting for the open-domain scenario. Each data instance is composed of a natural language question q, 10 candidate wiki documents, each document has its title and the content. The aim of multi-hop qa retriever is to derive a grounded document.
2. [Wikihop](https://qangaroo.cs.ucl.ac.uk/).
   As this dataset only contains the input NL query and grounded answer to the query. We generate the grounded documents by ourselves. According to its [paper](https://transacl.org/ojs/index.php/tacl/article/viewFile/1325/299), they provided a set of support documents, which contain the exact clue to fetch the answer. We generate the hop grounded document by the subject entity in the question.
   
    Specifically, based on the paper's introduction of the dataset construction, we generate the hop 1 grounded document by the entity linking tool. As the reasoning path to the query-answer, the start is the document which the subject entity in the query linked to, i.e., the intro page for the entity. The path tranversed based on the hyperlink structure in the wiki documents, and stop at the document which contains the answer entity. As there exists many documents which contains the answer entity, it is not possible to derive the ground supervision for hop i (i>1) retriever. Thus, we conduct our experiments on this [dataset](https://drive.google.com/drive/folders/1eDxVwc7BGPcYYXHSRyf2UuR2mUZm1OXz?usp=sharing).
  
    
