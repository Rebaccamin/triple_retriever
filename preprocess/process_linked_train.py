'''
this code is to process the linked para in the train data.
For each instance in train data, it has 50 candidate paras as the hop 1,
by hyperlinked paras expanded based on wiki, the candidate paras can be enlarged.

this is to process the expanded para, extract triple
'''
import pickle
import os
from tqdm import tqdm
from sqlitedict import SqliteDict


if not os.path.exists("./wiki_paras"):
  os.makedirs("./wiki_paras")

def getExist(dirname):
    filenames=[]
    for dir,_,files in os.walk(dirname):
        for i in files:
            if '.pkl' in i:
                filenames.append(os.path.join(dir,i))
    print('load %d files from the dir: %s...'%(len(filenames),dirname))
    return filenames

def loadfpkl(filename):
    if os.path.exists(filename) and '.pkl' in filename:
        print('load data %s.'%filename)
    else:
        print('the file %s not exists.'%filename)
        return

    with open(filename,'rb')as f:
        data=pickle.load(f)
    return data

def store2db(data,dbname):
    assert type(data)==dict
    db = SqliteDict(os.path.join('../wiki_paras', dbname+'.db'), autocommit=True)
    for i in data:
        db[i]= data[i]
    db.close()

    return

def store2pkl(data,filename):
    with open(filename,'wb')as f:
        pickle.dump(data,f)
    return

def find_linked(data):
    '''
    :param data: the training data, keys: context, all_linked_paras_dic, all_linked_para_title_dic (title: linked title; ...)
    :return:
    '''
    linked_paras_data=dict()
    for ins in tqdm(data):
        if len(ins['context'])<50:
            finished=ins['context']
            linked_dict=ins['all_linked_para_title_dic']
            linked_paras=[]
            for title in finished:
                if title in linked_dict:
                    linked_paras.extend(linked_dict[title])

            linked_paras=list(set(linked_paras))

            for para in linked_paras:
                if para in ins['all_linked_paras_dic']:
                    linked_paras_data[para]=ins['all_linked_paras_dic'][para]

    return linked_paras_data



def main():
    dirname = '../data/pickle_dir'
    filenames = getExist(dirname)

    dbname='linked_para_train'
    for file in filenames:
        data=loadfpkl(file)
        linked=find_linked(data)
        store2db(linked,dbname)


if __name__ == '__main__':
    main()