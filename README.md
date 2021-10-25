# triple_retriever

1. Pre-process. 
   We apply two existing OpenIE tools, [MinIE](https://github.com/uma-pi1/minie#minie-open-information-extraction-system) and [OpenIE]( https://stanfordnlp.github.io/CoreNLP/openie.html) to do the triple extraction over the text corpus. We first conduct [correference resolution](https://github.com/huggingface/neuralcoref) over the raw documents and then split the documents by NLTK sentence tokenizer into a set of sentences.

2. triple extractor.
3. retriever.


## dataset:
1. hotpotQA.
2. Wikihop.
   As this dataset is 
