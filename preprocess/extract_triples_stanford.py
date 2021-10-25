import os
import json
from standford import stanfordT,split2sen
import string
from tqdm import tqdm
import spacy
import itertools
import re
'''
extract triples by stanford open ie.
'''
class Preprocess():
    def __init__(self,file,filenames,dir):
        self.file=file
        self.filenames=filenames
        self.dir=dir
        self.nlp = spacy.load('en')



    def stanford_extract_para_ground(self,document,title,port):
        sens = split2sen(document)
        newsens = []
        for sen in sens:
            newsen= ' '.join(word.strip(string.punctuation) for word in sen.split())
            newsen = newsen.replace('\'s','')
            newsen = self.preprocess_sen_4(newsen)
            newsens.append(newsen)
        t = []
        for sen in newsens:
            t.extend(self.stanford_extract_sen(sen,title,port))
        return t

    def preprocess_sen_2(self, sentence, only_sentence=False):
        ''' process the sentence contains typo.
        :param sentence:
        :return:
        '''

        sentence = sentence.replace('\'s','')
        sentence = ' '.join(word.strip(string.punctuation) for word in sentence.split())
        sentence = self.preprocess_sen_4(sentence)

        if only_sentence==True:
            return sentence

        try:
           t = stanfordT(sentence)
        except:
            t = []
        return t

    def preprocess_sen_5(self,sentence):
        '''
        remove the () and its content in the () format in the sentence.
        for example: Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century.
        after: "Arthur's Magazine  was an American literary periodical published in Philadelphia in the 19th century."
        :param sentence:
        :return: sentence without (), the things in ().
        '''
        s = re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]", "", sentence)
        p1 = re.compile(r'[(](.*?)[)]', re.S)
        removed = re.findall(p1, sentence)

        return s, removed

    def stanford_extract_sen(self,sentence,para_title,port):

        if para_title!=None:
            t0,sentence = self.generate_born_triple(sentence,para_title)# process the () and remove it. as the () can influence the extraction results.
        else:
            t0=[]

        sentence1 = self.preprocess_sen_4(sentence)
        try:
           t1 = stanfordT(sentence1,port)
        except:
            t1 = []

        sentence2=self.preprocess_sen_2(sentence,only_sentence=True)
        try:
            t2 = stanfordT(sentence2,port)
        except:
            t2 = []

        sentence3,removed = self.preprocess_sen_5(sentence)
        try:
            t3 = stanfordT(sentence2,port)
        except:
            t3 = []

        k = t0+t1+t2+t3
        k.sort()
        t=list(k for k,_ in itertools.groupby(k))# de-redundancy

        return t,removed

    def preprocess_sen_4(self,sentence):
        '''
        remove the non-ascii characteristic.
        :param sentence:
        :return:
        '''
        printable = set(string.printable)
        sentence = ''.join(filter(lambda x: x in printable, sentence))
        return sentence

    def getExist(self,dir):
        filenames=[]
        for dir,_,files in os.walk(dir):
            for i in files:
                filenames.append(os.path.join(dir,i))
        return filenames

    def read(self,filename):
        with open(filename,'r')as f:
            data=json.load(f)
        return data


    def T_question(self,data):
        triples=[]
        if 'question' in data:
            if 'minie_t' in data['question']:
                triples+=data['question']['minie_t']
            if 'stanford_t' in data['question']:
                triples+= data['question']['stanford_t']
        return triples

    def T_data(self, data):
        triples=[]
        if 'minie_t' in data:
            triples+=data['minie_t']
        if 'stanford_t' in data:
            triples+=data['stanford_t']
        return triples

    def T_context(self,data):
        triples=[]
        if 'context' in data:
            context=data['context']
            for key in context:
                if 'minie_t' in context[key]:
                    triples+=context[key]['minie_t']
                if 'stanford_t' in context[key]:
                    triples+=context[key]['stanford_t']
        return triples

    def count_no_question_triple_instance(self,file):
        if os.path.exists(file):
            print('start to do data analysis of %.' % file)
        else:
            print('do not find the file %s.' % file)
            return
        data = self.read(file)

        number_q_no = 0
        for ins in data:
            t = self.T_question(ins)
            if len(t) == 0:
                number_q_no = number_q_no + 1

        print('there are %d instances without question triples .' % number_q_no)
        print('*' * 20)
        return number_q_no

    def processq(self, port, outputdir):
        files=self.getExist(self.dir)

        num=0
        print('Before: %d data without qt.'%len(files))
        print('start process question in the %s dir.'%self.dir)
        for file in tqdm(files):
            data=self.read(file)
            newdata=data
            qt = self.processsen(data['question']['text'],port)
            newdata['question']['stanford_t'] += qt
            if len(newdata['question']['stanford_t']+newdata['question']['minie_t'])==0:
                num = num+1
            if os.path.exists(outputdir)!=True:
                os.mkdir(outputdir)
            with open(os.path.join(outputdir,file.split('/')[-1]),'w')as f:
                json.dump(newdata,f,indent=4)
        print('After: %d data without qt.' % len(files))
        return

    def processsen(self,sentence,port):
        newsentence=''
        if sentence[-1]=='?':
            newsentence=sentence.replace('?','.')

        t = stanfordT(newsentence,port)
        if len(t)==0:
            if 'Which' in newsentence[:5]:
                newsentence=newsentence.replace('Which','')
                t = stanfordT(newsentence,port)
                return t
            else:
                return []
        else:
            return t

        # return newsentence

import argparse
def main():
    a = Preprocess(file='',filenames='',dir='')
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str)
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)
    parser.add_argument('--port',default=9000,type=int)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    file = args.file
    out_start=int(file.split('_')[2])
    if os.path.exists(file) and '.json' in file:
        data = a.read(file)
        newdata = []
        start = args.start
        end = min(args.end, len(data))
        for qa in tqdm(data[start:end]):
            newqa = qa
            context = qa['supports']
            newcontext = []
            for instance in context:
                text=instance['cor_text']
                triples=a.stanford_extract_para_ground(text,None,args.port)
                newinstance=instance
                newinstance['stanford_t']=triples
                newcontext.append(newinstance)
            newqa['supports'] = newcontext
            newdata.append(newqa)

        # dump the triple along the text to the output file.
        output_file='train_'+str(out_start+start)+'_'+str(out_start+end)+'.json'
        print('dump %d (src:%d) data to %s.' % (len(newdata), len(data), output_file))
        with open('text_triples/'+output_file, 'w')as f:
            json.dump(newdata, f, indent=4)

    else:
        print('no file exist!')

if __name__ == '__main__':
    main()