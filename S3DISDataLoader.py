# *_*coding:utf-8 *_*
import os
from torch.utils.data import Dataset
import numpy as np
import h5py

classes = ['ceiling','floor','wall','beam','column','window','door','table','chair','sofa','bookcase','board','clutter']
class2label = {cls: i for i,cls in enumerate(classes)}

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    # print('f shape',f.shape)
    print('data shape',data.shape)
    print('label shape',label.shape)
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)

def recognize_all_data(test_area = 5):
    ALL_FILES = getDataFiles('./indoor3d_sem_seg_hdf5_data/all_files.txt')
    print('ALL_FILES is', ALL_FILES)
    room_filelist = [line.rstrip() for line in open('./indoor3d_sem_seg_hdf5_data/room_filelist.txt')]
    data_batch_list = []
    label_batch_list = []
    for h5_filename in ALL_FILES:
        data_batch, label_batch = loadDataFile(h5_filename)
        data_batch_list.append(data_batch)
        label_batch_list.append(label_batch)
    data_batches = np.concatenate(data_batch_list, 0)
    print('data_batches shape is',data_batches.shape)
    label_batches = np.concatenate(label_batch_list, 0)
    print('label_batches shape is',label_batches.shape)
    
    test_area = 'Area_' + str(test_area)
    train_idxs = []
    test_idxs = []
    # notice, area = 5 is a test set.
    for i, room_name in enumerate(room_filelist):
        if test_area in room_name:
            test_idxs.append(i)
        else:
            train_idxs.append(i)
    # this is a train dataset
    train_data = data_batches[train_idxs, ...]
    train_label = label_batches[train_idxs]
    # this is a test dataset
    test_data = data_batches[test_idxs, ...]
    test_label = label_batches[test_idxs]
    # print two dataset's size
    # train_data (16733, 4096, 9) train_label (16733, 4096)
    # test_data (6852, 4096, 9) test_label (6852, 4096)
    print('train_data shape is',train_data.shape,'train_label shape is',train_label.shape)
    print('test_data shape is',test_data.shape,'test_label shape is', test_label.shape)
    print('train_data is',train_data,'train_label is',train_label)
    print('test_data is',test_data,'test_label is', test_label)    
    return train_data,train_label,test_data,test_label


class S3DISDataLoader(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
