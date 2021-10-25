# Triple_retriever

This respository contains the complete code for our paper "Triple-Fact Retriever: An explainable reasoning retrieval model for multi-hop QA problem.".

## Pre-process. 
  
  We apply two existing OpenIE tools, [MinIE](https://github.com/uma-pi1/minie#minie-open-information-extraction-system) and [OpenIE]( https://stanfordnlp.github.io/CoreNLP/openie.html) to do the triple extraction over the text corpus. We first conduct [correference resolution](https://github.com/huggingface/neuralcoref) over the raw documents and then split the documents by NLTK sentence tokenizer into a set of sentences.
  
  For the hotpotQA data, as each documement instance is consisted of a title and its corresponding text，e.g, 
  >{"Barack Obama":"Barack Hussein Obama II is an American politician, author, and retired attorney who served as the 44th president of the United States from 2009 to 2017. A member of the Democratic Party, Obama was the first African-American president of the United States. He previously served as a U.S. senator from Illinois from 2005 to 2008 and as an Illinois state senator from 1997 to 2004. Outside of politics, Obama has published three bestselling books; Dreams from My Father (1995), The Audacity of Hope (2006) and A Promised Land (2020)"}.
  
  We deploy a local correference resolution method by the title. Details in **extract_triples_minie.py and extract_triples_stanford.py**

## triple extractor

## retriever


## dataset:
1. hotpotQA.
2. Wikihop.
   As this dataset is 
