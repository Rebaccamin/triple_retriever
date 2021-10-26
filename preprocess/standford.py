import os
import time
import pprint
import random
import json
import sys
from tqdm import tqdm


def read(filename):
        with open(filename,'r')as f:
                data=json.load(f)
        return data


def stanford_api(sen,port=9000):

        shell = ''' curl --data '%s' '''  %(sen)+ ''' 'http://localhost:%d'''%port+'''/?properties={%22annotators%22%3A%22tokenize%2Cssplit%2Cpos%2Copenie%22%2C%22outputFormat%22%3A%22json%22}' -o - '''
        #print(shell)
        #print(shell)
        result_str = os.popen(shell).read()
        #print(result_str)
        #print(type(result_str))
        # time.sleep(random.uniform(1.5,10.5))
        return result_str

def parse(output):
   result=[]
   data=json.loads(output)
   if "sentences" in data:
        data=data["sentences"][0]
        if "openie" in data:
           result=data["openie"]
   return result

def fetchT(result):
        triples=[]
        for i in result:
                triple=[i['subject'], i['relation'], i['object']]
                triples.append(triple)
        return triples

def processSen(sen):
    if '\'' in sen:
        sen=sen.replace()

def stanfordT(sen,port):
   try:
           result_str = stanford_api(sen,port)
           result = parse(result_str)

           triple=fetchT(result)

           return triple
   except:
        return []

import nltk
from nltk.tokenize import WordPunctTokenizer
def split2sen(paragraph):
    paragraph=str(paragraph)
    sen_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle') 
    sentences = sen_tokenizer.tokenize(paragraph)
    return sentences

def main():
        filename=sys.argv[1]
        if os.path.isfile(filename):
                print('start process %s.'%filename)
        else:
                return
        data=read(filename)
        newdata=[]
        for i in tqdm(data):
                newi=i
                question=i['question']
                qt=[]  
                #if len(question['minie_t'])==0:
                sen=question['text']
                qt=stanfordT(sen)
                newi['question']['stanford_t']=qt

                context=i['context']
                newcontext=dict()
                for c in context:
                        text=context[c]['text']
                        if type(text)==list:
                                text=' '.join(text)
                        sens=split2sen(text)
                        triple=[]
                        for sen in sens:
                                triple.extend(stanfordT(sen))
                        newcontext[c]=context[c]
                        newcontext[c]['stanford_t']=triple
                newi['context']=newcontext
                newdata.append(newi)
        assert len(newdata)==len(data)
        outputfile = filename.replace('.json','_stantri.json')
        print('store the data to %s.'%outputfile)
        with open(outputfile,'w')as f:
                json.dump(newdata,f,indent=4)



        
       

if __name__ =="__main__":
    main()
    #sen='skdj'
    #stanford_api(sen)
