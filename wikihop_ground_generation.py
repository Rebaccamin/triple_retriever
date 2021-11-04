import json
from wikidataintegrator import wdi_core
import wikipediaapi
import nltk
import logging
import os
import argparse
from tqdm import  tqdm
import pickle
import string
import re
import wikipedia


class Wikihop():
    def __init__(self,logger):
        self.logger=logger
        self.name=''
        self.wiki_wiki = wikipediaapi.Wikipedia('en')

    def read(self,filename):
        print('load data from ' + filename)
        if '.json' in filename:
            with open(filename, 'r')as f:
                data = json.load(f)
            return data
        elif '.pkl' in filename:
            with open(filename, 'rb')as f:
                data = pickle.load(f)
            return data
        elif '.txt' in filename:
            with open(filename, 'r')as f:
                data = f.readlines()
            return data
        else:
            print('cannot read data from the file!')
            return
        return

    def detect_entity(self,query):
        tokens=query.split(' ')
        term=''
        for w in tokens:
            if '_' in w or w in ["genre", "developer", "screenwriter"]:
                continue
            else:
                term=term+w+' '
        return term.rstrip(' ')

    def assert_two_sets(self,set1,set2):
        if len(set1)<=10:
            return False
        if len(set2)<=10:
            return False

        if set(set1).issubset(set(set2)) or set(set2).issubset(set(set1)):
            return True
        elif len(set(set1).intersection(set(set2)))>len(set(set1))*0.9 and len(set(set1).intersection(set(set2)))>len(set(set2))*0.9:
            return True
        elif len(set(set1).intersection(set(set2)))>len(set(set1).difference(set(set2))):
            return True

        return False

    def set_words(self,document):
        sens=nltk.sent_tokenize(document)
        words=[]
        for sen in sens:
            #remove punctuation
            opt = re.sub(r'[^\w\s]', '', sen)
            words.extend(nltk.word_tokenize(opt))
        return words

    def obtain_ground_hop1_doc(self,data,file):
        newdata=[]
        for idx,instance in enumerate(tqdm(data)):
            query=instance['query']
            subject=self.detect_entity(query)
            supports=instance['supports']

            result = wdi_core.WDItemEngine.get_wd_search_results(subject, max_results=10)

            generated=False
            documents=[]

            for e in result:
                try:
                    wikidata_item = wdi_core.WDItemEngine(wd_item_id=e)
                    re = wikidata_item.get_wd_json_representation()

                    name = re['labels']['en']['value']

                    page_py = self.wiki_wiki.page(name)
                    text=page_py.text.lower()
                    text_words = self.set_words(text)
                    for doc in supports:
                        t=doc['text'].lower()
                        t_words=self.set_words(t)

                        if self.assert_two_sets(text_words,t_words):
                            documents.append(doc)
                            #negative documents
                            documents.append(supports.remove(doc))
                            newinstance = instance
                            newinstance['supports'] = documents
                            generated=True
                            break

                    if generated==True:
                        break
                except:
                    logging.info(subject+' '+e+'\n')

            if generated==False:
                documents=[]
                for doc in supports:
                    text=doc['text'].lower()
                    sen=nltk.sent_tokenize(text)[0]
                    words=nltk.word_tokenize(sen)
                    subject_words=nltk.word_tokenize(subject)
                    if set(subject_words).issubset(set(words)):
                        documents.append(doc)
                        # negative documents
                        documents.append(supports.remove(doc))
                        generated = True
                        newinstance = instance
                        newinstance['supports'] = documents
                        break

            if generated==False:
                newinstance=self.fixed_ground_wikipedia(instance)
                if newinstance!=None:
                    generated=True

            if generated==False:
                newinstance=self.fixed_groud_hop1(instance)
                if newinstance!=None:
                    generated=True


            if generated==False:
                with open('hop1_not_found_wiki/'+file.split('/')[-1].replace('.json','Num_'+str(idx)+'.json'),'w')as f:
                    json.dump(instance,f,indent=4)
                print('No %d data has problem.' %idx)
                #newinstance = self.fixed_groud_hop1(instance)
            else:
                newdata.append(newinstance)
        return newdata


    def generation_one_hop(self,dir,start,end, outdir='ground_hop1_wikihop'):
        filenames=[]
        if os.path.exists(outdir)!=True:
            os.mkdir(outdir)

        for dirname,_,files in os.walk(dir):
            for i in files:
                filenames.append(os.path.join(dirname,i))

        end=min(len(filenames),end)

        for file in filenames[start:end]:
            data=self.read(file)
            newdata=self.obtain_ground_hop1_doc(data,file)
            file1=file.replace(dir,outdir)

            print('dump %d training data into file: %s for hop 1 from %d data in file: %s .'%(len(newdata),file1,len(data),file))
            with open(file1,'w') as f:
                json.dump(newdata,f,indent=4)


    def fixed_groud_hop1(self,data):
        '''
        find the maximum words overlap document to the subject entity in the query as the ground hop 1 document.
        :param data:
        :return:
        '''
        query=data['query']
        title=self.detect_entity(query)
        wtitle=nltk.word_tokenize(title.lower())
        supports=data['supports']
        original_supports=supports
        ground=None
        max_overlap=0
        for doc in supports:
            try:
                text=doc['text'].lower()
                sen=nltk.sent_tokenize(text)[0]
                wtext=nltk.word_tokenize(sen)
                if len(set(wtext).intersection(set(wtitle)))>max_overlap:
                    max_overlap=len(set(wtext).intersection(set(wtitle)))
                    ground=doc
            except:
                return None
                print(json.dumps(doc,indent=4))

        if ground==None:
            return None

        documents=[]
        documents.append(ground)
        original_supports.remove(ground)
        documents.extend(original_supports)
        newdata=data
        newdata['suppports']=documents
        return newdata

    def fixed_ground_wikipedia(self,data):
        '''
        find more related documents based on the wikipedia api to the input query's entity.
        :param data:
        :return:
        '''
        try:
            query=data['query']
            title=self.detect_entity(query)
            supports=data['supports']
            documents=[]
            generated=False

            alist=wikipedia.search(title)
            for name in alist:
                page_py = self.wiki_wiki.page(name)
                text = page_py.text.lower()
                text_words = self.set_words(text)
                for doc in supports:
                    t = doc['text'].lower()
                    t_words = self.set_words(t)
                    if self.assert_two_sets(text_words, t_words):
                        documents.append(doc)
                        # negative documents
                        documents.append(supports.remove(doc))
                        generated = True
                        break
                if generated:
                    break

            if generated:
                newdata=data
                newdata['supports']=documents
                return newdata
            else:
                return None
        except:
            return None



def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dir',type=str)
    parser.add_argument('--outdir',type=str)
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--end',type=int,default=10)
    args=parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        filename='wikihop_generation_'+str(args.start)+'to'+str(args.end)+'.logs'
                        )

    logger = logging.getLogger(__name__)
    run=Wikihop(logger)
    run.generation_one_hop(args.dir,args.start,args.end,args.outdir)
    return

if __name__ == '__main__':
    main()