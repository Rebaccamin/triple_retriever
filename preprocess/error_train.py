import os
import json
import pickle
from tqdm import tqdm
''' process the training data.
specially, some instance: the ground para(s) has no triples.
'''
def getExist(dir):

    if os.path.exists(dir):
        filenames = []
        for dir, _, files in os.walk(dir):
            for file in files:
                filenames.append(os.path.join(dir, file))
        print('*****load %d files from the directory %s.*****' % (len(filenames), dir))
        return filenames
    else:
        print('no dir found!')
        return []

def read(file):
    if os.path.exists(file):
        # print('read data from %s.'%str(file))
        if '.json' in file:
            with open(file, 'r')as f:
                data = json.load(f)

                return data
        elif '.txt' in file:
            with open(file, 'r')as f:
                data = f.readlines()
                return data
        elif '.pkl' in file:
            with open(file, 'rb')as f:
                data = pickle.load(f)
            return data
    else:
        print('file not found!')
        return

def detect_no_ground(data):
    loss_ground = dict() # detect the ground para without triples.
    ins_ground_missed=[]#the ground paras not included in the 50 contexts.

    newdata=[]

    for ins in tqdm(data):
        short_gold=ins.short_gold
        error=0
        for title in short_gold:
            if title in ins.context:
                if len(ins.context[title]['merge_triples'])==0:
                    error=1
                    loss_ground[title]=ins.context[title]
            else:
               error=1
               ins_ground_missed.append(ins)
        if error==0:
            newdata.append(ins)


    return loss_ground,ins_ground_missed,newdata

def main():
    dirname='data/pickle_dir1'
    filenames=getExist(dirname)
    total_ground_no=dict()
    total_ins_ground_missed=[]
    for file in (filenames):
        print('***start process file %s.***'%file)
        data = read(file)
        loss_ground, ins_ground_missed,newdata =detect_no_ground(data)
        total_ground_no.update(loss_ground)
        total_ins_ground_missed.extend(ins_ground_missed)

        with open(file.replace('pickle_dir1','pickle_dir2'),'wb')as f:
            pickle.dump(newdata,f)
        print('before there are %d instances, now there are %d instances.'%(len(data),len(newdata)))

    with open('pickle_dir1_ground_no_triples.json','w')as f:
        json.dump(total_ground_no,f,indent=4)
    with open('pickle_dir1_ground_not_included.pkl','wb')as f:
        pickle.dump(total_ins_ground_missed,f)


    return


if __name__ == '__main__':
    main()