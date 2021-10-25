'''
this code is to process the train data (question) use stanford again.
'''
import os
import pickle
from tqdm import tqdm
from standford import stanfordT
import argparse


def getfiles(dirname):
    filenames=[]
    for dir,_,files in os.walk(dirname):
        for i in files:
            filenames.append(os.path.join(dir,i))
    return filenames

def read(filename):
    assert 'pkl' in filename
    with open(filename,'rb')as f:
        data=pickle.load(f)
    return data

def removeQmask(sen):
    sen=sen.replace('?',".")
    return sen

def de_duplicate_list(sents):
    remove=[]
    for i in range(len(sents)):
        for j in range(i+1,len(sents)):
            a=sents[i]
            b=sents[j]
            if set(a).issubset(set(b)):
                remove.append(sents[i])
            elif set(b).issubset(set(a)):
                remove.append(sents[j])
    return list(set(sents)-set(remove))

def t2sen(ts):
    sents=[]
    for t in ts:
        sen = (' '.join(t))
        sen = sen+' .'
        sents.append(sen)
    return list(set(sents))

def stanford_again():
    parse = argparse.ArgumentParser()
    parse.add_argument('--start', type=int, default=0)
    parse.add_argument('--end', type=int, default=10)
    args = parse.parse_args()

    dirname = 'pickle_dir3'
    newdir = dirname + '_new'
    if os.path.exists(newdir):
        print('the finished dir exists!')
    else:
        os.mkdir(newdir)
    filenames = getfiles(dirname)
    start = args.start
    end = args.end

    for i in range(start, end):
        print('start process file %d.' % i)
        file = filenames[i]
        data = read(file)
        newdata = []
        for ins in tqdm(data):
            newins = ins
            question = removeQmask(ins.question)
            t = stanfordT(question, 9000)
            if len(t) == 0:
                newdata.append(newins)
            else:
                tsens = t2sen(t)
                original = ins.question_t
                original.extend(tsens)
                newins.question_t = list(set(original))

                newdata.append(newins)
        assert len(newdata) == len(data)
        with open(file.replace(dirname, newdir), 'wb')as f:
            pickle.dump(newdata, f)
        print('finish store data to %s.' % file.replace(dirname, newdir))



def main():
    dirname = 'pickle_dir3_new'
    outdir='processed_pickle'
    if os.path.exists(outdir):
        print('outdir exists!')
        return
    else:
        os.mkdir(outdir)

    filenames=getfiles(dirname)
    qt_out_of_size=[]
    qt_zero=[]
    qt_12=[]
    for file in filenames:
        print('start process file %s.'%file)
        data=read(file)
        for ins in tqdm(data):
            if len(ins.question_t)==0:
                qt_zero.append(ins)
            elif len(ins.question_t)>2:
                qt_out_of_size.append(ins)
            else:
                qt_12.append(ins)
    print('there are %d question no triples.'%len(qt_zero))
    print('there are %d question with more than 2 triples.'%len(qt_out_of_size))
    print('there ard %d question has 1-2 triples.'%len(qt_12))
    with open(os.path.join(outdir,'common_train_ins.pkl',),'wb')as f:
        pickle.dump(qt_12,f)
    with open(os.path.join(outdir,'no_qt_train_ins.pkl',),'wb')as f:
        pickle.dump(qt_zero,f)
    with open(os.path.join(outdir,'more_qt_train_ins.pkl',),'wb')as f:
        pickle.dump(qt_out_of_size,f)










if __name__ == '__main__':
    main()