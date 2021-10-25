import json
import nltk
import logging
import pickle
from tqdm import tqdm
import argparse
import string
import os

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO

                        )

logger = logging.getLogger(__name__)

class Merge():
    def __init__(self):
        self.name = 'merge'

    def read(self,filename):
        logger.info('load data from ' + filename)
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

    def cor_resolu_triple(sentences, title):
        newsentences = []
        string_pun = string.punctuation
        for sentence in sentences:
            newsen = []

            cor = 0
            if type(sentence) == list:
                sentence = ' '.join(sentence)
            # cor resolution by nltk, process daici.
            words = nltk.word_tokenize(sentence)
            pos_tags = nltk.pos_tag(words)
            for word, tag in pos_tags:
                if tag == 'PRP' or tag == 'WP':  # He,she, it; who what
                    cor = 1
                    newsen.append(title)
                else:
                    newsen.append(word)
            newsen = ' '.join(newsen)

            # detect the incomplete entity representation.
            if cor == 0:
                tempa = title
                for i in string_pun:
                    tempa = tempa.replace(i, '')
                tempb = newsen
                for i in string_pun:
                    tempb = tempb.replace(i, '')
                inters = set(tempa.split(" ")).intersection(set(tempb.split(" ")))

                con_flag = -1
                last_location = -1
                current_location = -1
                for x in inters:
                    if x in newsen and x not in ['for', 'of', 'to'] and len(x) > 3:
                        current_location = newsen.find(x)
                        if last_location == -1 and con_flag == -1:
                            last_location = current_location
                            if title not in newsen:
                                newsen = newsen.replace(x, title, 1)
                            con_flag = 0
                        else:
                            if abs(last_location - current_location) == 1 and con_flag == 0:
                                newsen = newsen.replace(x, '')
                            else:
                                break

            # as the matching of entity can induce redundancy of entities in a triple.

            # sentence = newsen.lower()
            newsentences.append(newsen)
        return newsentences

    def merge_triple_sen(self, triples):
        '''
        merge the triples: if one triple can be seen as a part of another triple, it is a redundancy.
        we will remove the sub triple from the whole triple sets.
        '''
        remove = []
        for i in range(len(triples)):
            for j in range(i + 1, len(triples)):
                alist = triples[i]
                blist = triples[j]

                if set(alist).issubset(set(blist)) :
                    remove.append(alist)
                elif set(blist).issubset(set(alist)):
                    remove.append(blist)
        return list(set(triples) - set(remove))

    def de_redundancy_triple(self, triples):
        removed = []
        for i in range(len(triples)):
            for j in range(i + 1, len(triples)):
                words_1 = nltk.word_tokenize(' '.join(triples[i]))
                words_2 = nltk.word_tokenize(' '.join(triples[j]))
                if set(words_1).issubset(set(words_2)):
                    removed.append(triples[j])
                elif set(words_2).issubset(set(words_1)):
                    removed.append(triples[j])
        return list(set(triples) - set(removed))

    def lists_2_list(self,lists):
        triples=[]
        for t in lists:
            if len(t)!=0:
                triples.extend(t)
        return triples

    def process_triples(self,dir,outdir):
        '''
        convert the lists of triples to a list of triple.
        :param filename:
        :return:
        '''
        filenames=[]
        for dirname,_,files in os.walk(dir):
            for f in files:
                filenames.append(os.path.join(dirname,f))
        if os.path.exists(outdir)!=True:
            os.mkdir(outdir)
        for filename in filenames:
            data=self.read(filename)
            newdata=[]

            for qa in tqdm(data):
                supports=qa['supports']
                newqa=qa
                newsupports=[]
                for instance  in supports:
                    minie_t=instance['minie_t']
                    minie_t=self.lists_2_list(minie_t)
                    newinstance=instance
                    newinstance['minie_t']=minie_t
                    stanford_t=instance['stanford_t']
                    stanford_t=self.lists_2_list(stanford_t)
                    newinstance['stanford_t']=stanford_t
                    newsupports.append(newinstance)
                newqa['supports']=newsupports
                newdata.append(newqa)

            filename=filename.replace(dir,outdir)
            with open(filename,'w')as f:
                json.dump(newdata,f,indent=4)
        return

    def merge_triples(self,dir):
        filenames = []
        for dirname, _, files in os.walk(dir):
            for f in files:
                filenames.append(os.path.join(dirname, f))
        for filename in filenames:
            data=self.read(filename)
            newdata=[]
            for qa in data:
                supports=qa['supports']
                newsupports=[]
                for instance in tqdm(supports):
                    t0=instance['minie_t']
                    t1=instance['stanford_t']
                    t= (t0+t1)
                    # if hotpotQA, correference by its title
                    # title=instance    supports is a dict.
                    # t0 =supports[instance]['minie_t']
                    # t1 =supports[instance]['stanford_t']
                    # t =self.cor_resolu_triple(t,title)
                    t = self.de_redundancy_triple(t)
                    newinstance=instance
                    newinstance['merge_t']=t
                    newsupports.append(newinstance)
                newqa=qa
                newqa['supports'] = newsupports
                newdata.append(newqa)
            with open(filename,'w')as f:
                json.dump(newdata,f,indent=4)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dir',type=str)
    parser.add_argument('--outdir',type=str)
    args=parser.parse_args()
    runner=Merge()
    runner.process_triples(args.dir,args.outdir)
    runner.merge_triples(args.outdir)




if __name__ == '__main__':
    main()