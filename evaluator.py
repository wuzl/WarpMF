from utils import free_memory
import glob
import numpy as np
import os

class Evaluator(object):
    def __init__(self, valid_path, start, end):
        print ('Before init evaluator: {m}'.format(m = free_memory()))
        self.valid_data = {}
        for fn in glob.glob(os.path.join(valid_path, '*')):
            for line in open(fn, 'r'):
                value = line.rstrip('\r\n').split('\t')
                if int(value[0]) < start or int(value[0]) > end:
                    continue
                self.valid_data[int(value[0]) - start] = (set(map(int, value[1].split('|'))), set(map(int, value[2].split('|'))))
        self.valid_user = list(self.valid_data.keys())
        print ('After init evaluator: {m}'.format(m = free_memory()))

    def users(self):
        return self.valid_user

    def eval(self, user_tops, k, task='precision'):
        metrics = []
        for user_id, tops in user_tops:
            top_n_items = 0
            hits = 0
            for i in tops:
                # ignore item in the training set
                if i in self.valid_data[user_id][1]:
                    continue
                elif i in self.valid_data[user_id][0]:
                    hits += 1
                top_n_items += 1
                if top_n_items == k:
                    break
            if task == 'recall':
                metrics.append(hits / float(len(self.valid_data[user_id][0])))
            else:
                metrics.append(hits / float(k))
        return np.mean(metrics)
