import os
from process import triple_extraction_minie,split2sen
import json
from tqdm import tqdm
import spacy
import neuralcoref
import string
import math
import re
import itertools
from multiprocessing import Pool
import unicodedata



class Preprocess():
    def __init__(self,dir):
        self.dir=dir
        self.nlp = spacy.load('en')
        neuralcoref.add_to_pipe(self.nlp)

    def remove_brackets(self,sen):
        '''
        remove the brackets in the sentences.
        :param sen:
        :return:
        '''
        if '(' in sen and ')' in sen:
            a = sen.index('(')
            b = sen.index(')')
            neiron = sen[a:b + 1]
            newsen = sen.replace(neiron, '')
            return newsen, 1
        return sen, 0

    def finderror(self, files, start):
        for idx, file in enumerate(files):
            if idx>start:
                data=self.read(file)
                t=self.preprocess_sen_1(data['question']['text'])
                if len(t)>0:
                    continue
                t=self.preprocess_sen_2(data['question']['text'])
                if len(t)>0:
                    continue
                t=self.preprocess_sen_3(data['question']['text'])
                if len(t)>0:
                    continue
                print(idx)
                print(file)
                return
        return


    def getExist(self, dir):
        filenames=[]
        if os.path.exists(dir):
            print('fetch all the data files from dir %s.'%dir)
        for dir, _, files in os.walk(dir):
            for i in files:
                filenames.append(os.path.join(dir,i))
        print('there are %d files loaded.'%len(filenames))
        return filenames

    def read(self,filename):
        with open(filename,'r')as f:
            data=json.load(f)
        return data

    def correference_resolution_doc(self, data):
        '''
        :param data: a raw document from wikipedia.
        :return: the document with the form of finished-correference-resolution. (only person), he/his/she/... has been replaced by the exact noun.
        '''
        if type(data)==list:
            data = ' '.join(data)
        doc = self.nlp(data)
        if doc._.has_coref:
            #print(doc._.coref_clusters)
            return doc._.coref_resolved
        else:
            return data

    def normalize(self,text):
        """Resolve different type of unicode encodings / capitarization in HotpotQA data."""
        text = unicodedata.normalize('NFD', text)
        return text[0].capitalize() + text[1:]

    def make_wiki_id(self,title, para_index):
        title_id = "{0}_{1}".format(self.normalize(title), para_index)
        return title_id

    def obtainPtitle(self, plist):
        result = []
        for i in plist:
            result.append(self.make_wiki_id(i[0], 0))
        return result

    def obtain_ground(self,ground,keyid):
        for data in ground:
            if data['_id']==keyid:
                ground_title = data['supporting_facts']
                ground_title = self.obtainPtitle(ground_title)
                return ground_title

        return None

    def coref_text(self, filenames, output, ground=None, extract=False):
        #filenames = self.getExist()

        for file in tqdm(filenames):
            #print(file)
            data = self.read(file)
            newdata=data
            if 'json' in newdata['q_id']:
                newdata['q_id']=newdata['q_id'].split('.')[0]
            newdata['question']['cor_text'] = self.correference_resolution_doc(newdata['question']['text'])

            if extract==True:
                newdata['question']['minie_t_cor'] = self.minie_extract_sen(newdata['question']['text'])

            if ground==None:
                new_context_dict = dict()
                new_context_dicts = dict()
                for key in data['context']:
                    new_context_dict = newdata['context'][key]
                    new_context_dict['cor_text'] = self.correference_resolution_doc(data['context'][key]['text'])
                    if extract==True:
                        new_context_dict['minie_t_cor'] = self.minie_extract_para(new_context_dict['cor_text'])
                    new_context_dicts[key]=new_context_dict

                newdata['context']=new_context_dicts
            else:
                if newdata['q_id'] in ground:
                     groundtitle= ground[newdata['q_id']]  #self.obtain_ground(ground,data['q_id'])
                else:
                    print('did not find the key id:%s.'%str(newdata['q_id']))
                    groundtitle = None
                new_context_dicts = dict()
                for key in data['context']:
                    if key in groundtitle:
                        new_context_dict = newdata['context'][key]
                        new_context_dict['cor_text'] = self.correference_resolution_doc(data['context'][key]['text'])
                        if extract==True:
                            new_context_dict['minie_t_cor'] = self.minie_extract_para_ground(new_context_dict['cor_text'])
                        new_context_dicts[key] = new_context_dict
                    else:
                        new_context_dict = newdata['context'][key]
                        new_context_dict['cor_text'] = self.correference_resolution_doc(data['context'][key]['text'])
                        if extract == True:
                            new_context_dict['minie_t_cor'] = []#self.minie_extract_para(new_context_dict['cor_text'])
                        new_context_dicts[key] = new_context_dict

                newdata['context'] = new_context_dicts

            assert len(new_context_dicts)==len(data['context'])
            assert list(new_context_dicts.keys())==list(data['context'].keys())

            if os.path.exists(output)!=True:
                os.mkdir(output)
            with open(os.path.join(output,file.split('/')[-1]),'w')as f:
                json.dump(newdata, f, indent=4)
        return

    def minie_extract_para_ground(self,document,title):
        sens = split2sen(document)
        newsens = []
        for sen in sens:
            newsen= ' '.join(word.strip(string.punctuation) for word in sen.split())
            newsen = newsen.replace('\'s','')
            newsen = self.preprocess_sen_4(newsen)
            newsen = self.remove_brackets(newsen)
            newsens.append(newsen)
        t = []
        for sen in newsens:
            t.extend(self.minie_extract_sen(sen,title))
        return t

    def minie_extract_para(self,document):
        '''
        :param document: the input already has been processed by correference resolution.
        :return: extracted triples for the input document.
        '''
        sens = split2sen(document)
        # newsens=[]
        # for sen in sens:
        #     newsens.append(' '.join(word.strip(string.punctuation) for word in sen.split()))
        t = []
        for sen in sens:
           t.extend(self.minie_extract_sen(sen))
        return t

    def preprocess_sen_1(self,sentence, change='.'):
        '''
        process the common question sentence which contain the question masks: remove it or replace it (e.g.: ... which people .. -> ... a people ...) by an/a.
        :param sentence:
        :param change:
        :return:
        '''
        sentence = self.preprocess_sen(sentence)
        sentence = self.preprocess_sen_4(sentence)

        whPattern = re.compile(r'(.*)who|what|how|where|when|why|which|whom|whose(\.*)', re.IGNORECASE)
        whWord = whPattern.search(sentence)
        if whWord:
            start=whWord.span()[0]
            if start>1:
                whQuestion = whWord.group()  # only detect the first question mask.
                sentence = sentence.replace(whQuestion, 'a')
            else:
                whQuestion = whWord.group()# only detect the first question mask.
                sentence=sentence.replace(whQuestion, '.')
        try:
            t = triple_extraction_minie(sentence)
        except:
            t = []
        return t

    def preprocess_sen_2(self, sentence, only_sentence=False):
        ''' process the sentence contains typo.
        :param sentence:
        :return:
        '''
        sentence = self.preprocess_sen(sentence)
        sentence = sentence.replace('\'s','')
        sentence = ' '.join(word.strip(string.punctuation) for word in sentence.split())
        sentence = self.preprocess_sen_4(sentence)

        if only_sentence==True:
            return sentence

        try:
           t = triple_extraction_minie(sentence)
        except:
            t = []
        return t

    def preprocess_sen_3(self,sentence):
        '''
        add human defined patterns.
         process some special question sentence case.
        :param sentence:
        :return: triples
        '''
        sentence = self.preprocess_sen(sentence)
        sentence = self.preprocess_sen_4(sentence)

        if 'What year' in sentence:
            sentence = self.preprocess_sen_2(sentence, only_sentence=True)
            sentence = sentence.lstrip('What year')
            sentence = sentence[:-1] + ' What year.'.lower()
        elif 'Who is older' in sentence:
            sentence = self.preprocess_sen_2(sentence, only_sentence=True)
            sentence = sentence.lstrip('Who is older')
            sentence = sentence[:-1] + ' is older.'.lower()
        elif 'In what year' in sentence:
            sentence = self.preprocess_sen_2(sentence, only_sentence=True)
            sentence = sentence.lstrip('In what year')
            sentence = sentence[:-1] + ' In what year.'.lower()
        # elif 'who' in sentence[5:]:
        #     sentence =

        try:
           t = triple_extraction_minie(sentence)
        except:
            t = []

        return t

    def preprocess_sen_4(self,sentence):
        '''
        remove the non-ascii characteristic.
        :param sentence:
        :return:
        '''
        printable = set(string.printable)
        sentence = ''.join(filter(lambda x: x in printable, sentence))
        return sentence

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

    def process_1_file(self,file,groudfile='supf_id.json'):
        data = self.read(file)
        ground = self.read(groudfile)
        ground_ps = ground[data['q_id']]
        newdata = data

        t = data['question']['minie_t']+data['question']['stanford_t']+data['question']['minie_t_cor']
        if len(t)==0:
            qt=self.preprocess_sen_3(data['question']['text'])
            if len(qt)==0:
                qt =self.preprocess_sen_2(data['question']['text'])
                if len(qt)==0:
                    qt=self.preprocess_sen_1(data['question']['text'])
            newdata['question']['minie_t_cor'] = qt


        newcontext = dict()
        for key in newdata['context']:
            if key in ground_ps:
                t = self.minie_extract_para_ground_incre1(newdata['context'][key]['cor_text'])
                newcontext[key] = newdata['context'][key]
                newcontext[key]['minie_t_cor'] = t
                print('*****Before: %d   Now: %d. *****'%(len(newdata['context'][key]['minie_t_cor']),len(t)))
            else:
                newcontext[key] = newdata['context'][key]
        newdata['context']=newcontext

        output = 'fixed_dir'
        if os.path.exists(output):
            with open(os.path.join(output,file),'w')as f:
               json.dump(newdata,f,indent=4)
        else:
            os.mkdir(output)
            with open(os.path.join(output,file),'w')as f:
               json.dump(newdata,f,indent=4)
        print('finished process file %s.'%file)
        return

    def minie_extract_para_ground_incre1(self, document):

        sens = split2sen(document)
        newsens = []
        for sen in sens:
            newsen= ' '.join(word.strip(string.punctuation) for word in sen.split())
            newsen = self.preprocess_sen_4(newsen)
            newsens.append(newsen)
        t = []
        for sen in newsens:
            t.extend(self.minie_extract_sen(sen))
        return t

    def preprocess_sen(self,sentence):
        if len(sentence)>0 and sentence[-1]=='?':
            sentence=sentence.replace('?','.')
        return sentence


    def generate_born_triple(self,sen,para_title):
        p1 = re.compile(r'[(](.*?)[)]', re.S)
        test = re.findall(p1, sen)
        triples=[]
        for i in test:
            if 'born' in i:
                location = i.find('born')
                triples.append([para_title,"born on",i[location+4:]])

        sen = re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]", "", sen)
        return triples,sen



    def minie_extract_sen(self,sentence,para_title):
        sentence = self.preprocess_sen(sentence)# wenju dao chenshuju
        if para_title!=None:
            t0,sentence = self.generate_born_triple(sentence,para_title)# process the () and remove it. as the () can influence the extraction results.
        else:
            t0=[]

        sentence1 = self.preprocess_sen_4(sentence)
        try:
           t1 = triple_extraction_minie(sentence1)
        except:
            t1 = []

        sentence2=self.preprocess_sen_2(sentence,only_sentence=True)
        try:
            t2 = triple_extraction_minie(sentence2)
        except:
            t2 = []

        sentence3,removed = self.preprocess_sen_5(sentence)
        try:
            t3 = triple_extraction_minie(sentence2)
        except:
            t3 = []

        k = t0+t1+t2+t3
        k.sort()
        t=list(k for k,_ in itertools.groupby(k))# de-redundancy

        return t,removed

    def process_q_minie(self,outputdir):
        filenames=self.getExist()
        no=0
        for file in tqdm(filenames):
            data=self.read(file)
            question=data['question']['text']
            question=self.preprocess_sen(question)
            t=triple_extraction_minie(question)
            newdata=data
            newdata['question']['minie_t']+=t
            if len(t)!=0:
                no=no+1

            if os.path.exists(outputdir)!=True:
                os.mkdir(outputdir)
            with open(os.path.join(outputdir,file.split('/')[-1]),'w')as f:
                json.dump(newdata,f,indent=4)
        print('there are %d data have triple (in total: %d).'%(no,len(filenames)))
        return

    def run(self,filenames, index, size, outputdir, grounddata, extracted=True):  # data:the input data，index:from which part on the input data ，size: the number of processer
        size = math.ceil(len(filenames) / size)
        start = size * index
        end = (index + 1) * size if (index + 1) * size < len(filenames) else len(filenames)
        temp_data = filenames[start:end]
        # do something
        self.coref_text(temp_data, output=outputdir, ground=grounddata, extract=extracted)

        return

    def coref_text_train(self,file):
        print('***start process file: %s.***'%file)
        data = self.read(file)
        newdata=[]
        for ins in tqdm(data):
            newins = ins
            newins['question']['cor_text'] = self.correference_resolution_doc(newins['question']['text'])
            newins['question']['minie_t_cor'] = self.minie_extract_sen(newins['question']['cor_text'])
            newins['question']['minie_t'] = self.minie_extract_sen(newins['question']['text'])

            newcontext=dict()
            for title in newins['context']:
                if title in newins['short_gold']:
                    newcontext[title] = newins['context'][title]
                    newcontext[title]['cor_text'] = self.correference_resolution_doc(newins['context'][title]['text'])
                    newcontext[title]['minie_t_cor'] = self.minie_extract_para_ground(newcontext[title]['cor_text'])
                else:
                    newcontext[title] = newins['context'][title]
                    newcontext[title]['cor_text'] = self.correference_resolution_doc(newins['context'][title]['text'])
                    newcontext[title]['minie_t_cor'] = []

            newins['context']=newcontext
            newdata.append(newins)

            assert len(newcontext)==len(newins['context'])

        assert len(newdata)==len(data)
        return newdata

    def run_process_train(self,filenames,index,output=None):
        file=filenames[index]

        newdata = self.coref_text_train(file)
        with open(file,'w')as f:
            json.dump(newdata, f, indent=4)
        print('finished reprocess the data of file: %s.'%file)
        return

    def parallel_process(self,dirname, output, groundfile, extract):
        filenames = self.getExist(dirname)

        print('there are %d files need to process.'%len(filenames))
        print('*'*10)

        processor = len(filenames)  #
        res = []
        p = Pool(processor)
        for i in range(processor):
            res.append(p.apply_async(self.run_process_train, args=(filenames, i, output)))
            print(str(i) + ' processor started !')
        p.close()
        p.join()

        outdata = []
        for i in res:
            outdata.extend(i.get())

import argparse

import nltk
def main():

    '''
    this is to process the eval data.
    extract triples from the text.

    for each para is a list of sentence. and i donot use correference resolution. instead, i detect the pronoun words(he,she..) and replace such words by the para title.
    :return:
    '''
    dev_file='hotpot_dev_fullwiki_v1.json'
    a = Preprocess(dir=' ')
    dev_data=a.read(dev_file)
    print('***start process file:%s.***'%dev_file)
    new_dev_data=[]
    for data in tqdm(dev_data):
        newdata = data
        context = data['context']
        newcontext=dict()
        for para in context:
            triples=[]
            para_title=para[0]
            para_text=para[1]
            if type(para_text)==list:
                for sen in para_text:
                    words = nltk.word_tokenize(sen)
                    pos_tags = nltk.pos_tag(words)
                    newsen = []
                    lastword=''
                    lasttag=''
                    for word, tag in pos_tags:
                        if tag == 'PRP' or tag == 'WP' or tag=='PRP$':  # He,she, it; who what
                             newsen.append(para_title)
                        else:
                            if tag=='NN' and lasttag=='DT': #e.g., replace "the film" by a sepecific film name.
                                newsen.remove(lastword)
                                newsen.append(para_title)
                            else:
                                newsen.append(word)

                        lastword=word
                        lasttag=tag

                    newsen = ' '.join(newsen)
                    triples.extend(a.minie_extract_sen(newsen,para_title))

            if type(para_text)==list:
                para_text = ' '.join(para_text)

            newcontext[para_title]={'text': para_text,'merge_triples':triples}
        newdata['context'] = newcontext
        new_dev_data.append(newdata)

    with open('hotpot_dev_fullwiki_extracted_triples.json','w')as f:
        json.dump(new_dev_data,f,indent=4)


def main1():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file',type=str)
    # a = Preprocess(dir='merge_files/')
    # a.process_1_file(args.file)
    parser.add_argument('--inputdir',type=str,required=True)
    parser.add_argument('--fileidx',type=int,default=-1)
    args=parser.parse_args()

    a = Preprocess(dir='hotpotqa_minie_stanford/')
    if args.fileidx!=-1:
        filenames = a.getExist(args.inputdir)
        a.run_process_train(filenames, args.fileidx)
    else:
        a.parallel_process(dirname=args.inputdir, output='hotpotqa_minie_stanford', groundfile='supf_id.json',extract=False)


def main2():
    '''process wikihop'''
    a=Preprocess(dir='')
    parser=argparse.ArgumentParser()
    parser.add_argument('--file',type=str)
    parser.add_argument('--start',type=int)
    parser.add_argument('--end',type=int)
    parser.add_argument('--output',type=str)
    args=parser.parse_args()

    file=args.file
    if os.path.exists(file) and '.json' in file:
        data=a.read(file)
        newdata=[]
        start=args.start
        end=min(args.end,len(data))
        for qa in tqdm(data[start:end]):
            newqa=qa
            context=qa['supports']
            newcontext=[]
            for text in context:
                cfr=a.correference_resolution_doc(text)
                newcontext.append({"text": text, "cor_text": cfr})
                #triples=a.minie_extract_para_ground(cfr,None)
                #newcontext.append({"text":text,"cor_text":cfr,"minie_t":triples})
            newqa['supports']=newcontext
            newdata.append(newqa)

        #dump the triple along the text to the output file.
        print('dump %d (src:%d) data to %s.'%(len(newdata),len(data),args.output))
        with open(args.output,'w')as f:
            json.dump(newdata,f,indent=4)

    else:
        print('no file exist!')



if __name__ == '__main__':
    main2()