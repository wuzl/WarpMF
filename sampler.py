from multiprocessing import Process, Queue
from utils import free_memory
import glob
import numpy as np
import os

user_to_positive_sets = {}

def sample_function(train_files, item2weight, negative_num, batch_size, result_queue, start, end):
    """
    :param train_files: file list of user-positive-item pairs data
    :param item2weight: weight of item
    :param negative_num: num of negative item
    :param batch_size: number of samples to return
    :param result_queue: the output queue
    :return: None
    """
    print (train_files)
    total_weight = sum(item2weight.values())
    item_list, weight_list = [], []
    for k, v in item2weight.items():
        item_list.append(k)
        weight_list.append(v/total_weight)
    user_positive_items_pairs = []
    negative_samples = np.random.choice(item_list, batch_size*negative_num, p=weight_list).reshape((batch_size, negative_num))
    negative_flags = np.ones_like(negative_samples, dtype=np.bool)
    file_batch = 0
    while True:
        np.random.shuffle(train_files)
        for fn in train_files:
            print (fn, file_batch)
            file_batch = 0
            for line in open(fn, 'r'):
                token = line.rstrip('\r\n').split('\t')
                value = [int(token[0]), int(token[1])]
                if value[0] < start or value[0] > end:
                    continue
                else:
                    value[0] = value[0] - start
                #https://stackoverflow.com/questions/68630/are-tuples-more-efficient-than-lists-in-python
                user_positive_items_pairs.append(tuple(value))
                count = len(user_positive_items_pairs)
                positive_set = user_to_positive_sets[value[0]]
                for index, item in enumerate(negative_samples[count-1]):
                    if item in positive_set:
                        negative_flags[count-1][index] = False
                if count == batch_size:
                    result_queue.put((user_positive_items_pairs, negative_samples, negative_flags))
                    user_positive_items_pairs = []
                    negative_samples = np.random.choice(item_list, batch_size*negative_num, p=weight_list).reshape((batch_size, negative_num))
                    negative_flags = np.ones_like(negative_samples, dtype=np.bool)
                    file_batch += 1

class WarpSampler(object):
    """
    A generator that, in parallel, generates tuples: user-positive-item pairs, negative-items
    of the shapes (Batch Size, 2) and (Batch Size, N_Negative)
    """
    def __init__(self, train_path, batch_size=10000, n_workers=5, item2weight={}, negative_num = 100, start=0, end=0):
        print ('before init sampler: {m}'.format(m = free_memory()))
        for fn in glob.glob(os.path.join(train_path, '*')):
            for line in open(fn, 'r'):
                token = line.rstrip('\r\n').split('\t')
                value = [int(token[0]), int(token[1])]
                if value[0] < start or value[0] > end:
                    continue
                else:
                    value[0] = value[0] - start
                if value[0] not in user_to_positive_sets:
                    user_to_positive_sets[value[0]] = set()
                user_to_positive_sets[value[0]].add(value[1])
        self.result_queue = Queue(maxsize=n_workers*2)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(Process(target=sample_function, args=([f for f in glob.glob(os.path.join(train_path, '*')) if (hash(f)%n_workers) == i],
                                                                         item2weight, negative_num, batch_size, self.result_queue, start, end)))
            self.processors[-1].start()
        print ('After init sampler: {m}'.format(m = free_memory()))

    def next_batch(self):
        print ('Queue data: %s' % self.result_queue.qsize())
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
