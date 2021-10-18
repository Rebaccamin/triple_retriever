import pickle


def split2smaller():
    files='data/pickle_dir2'# split the large data file to smaller ones.
    num = 0
    for file in files:
        with open(file, 'rb')as f:
            data = pickle.load(f)
        splits = len(data) // 1000

        assertion = 0
        for i in range(splits + 1):
            start = i * 1000
            end = start + 1000
            if i == splits:
                end = min(end, len(data))
            tempdata = data[start:end]
            assertion += len(tempdata)
            with open('data/pickle_dir3/train_file_' + str(num) + '.pkl', 'wb')as f:
                pickle.dump(tempdata, f)
            num = num + 1
        assert assertion == len(data)
    return