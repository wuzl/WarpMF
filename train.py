import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from utils import free_memory, chunks
from tensorflow.python.client import timeline
from sampler import WarpSampler
from evaluator import Evaluator
import glob


class WarpMF(object):
    def __init__(self,
                 n_users,
                 n_items,
                 embed_dim=20,
                 margin=1.5,
                 master_learning_rate=0.1,
                 alpha=0.05,
                 ):
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim

        self.alpha = alpha
        self.margin = margin

        self.master_learning_rate = master_learning_rate

        self.user_positive_items_pairs = tf.placeholder(tf.int32, [None, 2])
        self.negative_samples = tf.placeholder(tf.int32, [None, None])
        self.negative_flags = tf.placeholder(tf.float32, [None, None])
        self.score_user_ids = tf.placeholder(tf.int32, [None])

        self.item_bias = tf.Variable(tf.random_normal([self.n_items, 1], stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32))
        self.user_embeddings = tf.Variable(tf.random_normal([self.n_users, self.embed_dim], stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32))
        self.item_embeddings = tf.Variable(tf.random_normal([self.n_items, self.embed_dim], stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32))

        self._build_graph()


    def _build_graph(self):
        # Let
        # N = batch size,
        # K = embedding size,
        # W = number of negative samples per a user-positive-item pair

        # user embedding (N, K)
        users = tf.nn.embedding_lookup(self.user_embeddings, self.user_positive_items_pairs[:, 0], name="users")
        user_reg = tf.reduce_sum(tf.square(users), 1, name="user_reg")

        # positive item embedding (N, K)
        pos_items = tf.nn.embedding_lookup(self.item_embeddings, self.user_positive_items_pairs[:, 1], name="pos_items")
        pos_reg = tf.reduce_sum(tf.square(pos_items), 1, name="pos_reg")
        pos_bias = tf.squeeze(tf.nn.embedding_lookup(self.item_bias, self.user_positive_items_pairs[:, 1], name="pos_bias"))
        # positive item to user distance (N)
        pos_distances = tf.reduce_sum(tf.multiply(users, pos_items), 1 ) + pos_bias

        # negative item embedding (N, K, W)
        neg_items = tf.transpose(tf.nn.embedding_lookup(self.item_embeddings, self.negative_samples), (0, 2, 1), name="neg_items")
        neg_reg = tf.reduce_sum(tf.square(neg_items), 1, name="neg_reg")
        neg_bias = tf.squeeze(tf.nn.embedding_lookup(self.item_bias, self.negative_samples), name="neg_bias")
        # distance to negative items (N x W)
        distance_to_neg_items = tf.reduce_sum(tf.multiply(tf.expand_dims(users, -1), neg_items), 1) + neg_bias

        impostors = tf.multiply(self.negative_flags, (tf.expand_dims(-pos_distances, -1) + distance_to_neg_items + self.margin))
        indexes = tf.where(tf.greater(impostors, 0))

        self.impostor_num = tf.shape(indexes)[0]
        self.impostor_log = tf.nn.moments(impostors, axes=[0, 1])

        x_min_y = tf.segment_min(indexes[:, 1], indexes[:, 0])
        uni_x, _ = tf.unique(indexes[:,0])
        uni_y = tf.nn.embedding_lookup(x_min_y, uni_x)
        xy = tf.concat([tf.expand_dims(uni_x, -1), tf.expand_dims(uni_y, -1)], 1)

        impostor_xy = tf.gather_nd(impostors, xy)
        rank = tf.log((self.n_items - 1) / tf.cast(uni_y + 1, tf.float32))

        self.eloss = tf.reduce_sum(tf.clip_by_value(rank * impostor_xy, 0, 10))
        self.rloss = tf.reduce_sum(self.alpha * (tf.gather_nd(neg_reg, xy) + tf.nn.embedding_lookup(pos_reg, uni_x) + tf.nn.embedding_lookup(user_reg, uni_x)))

        self.loss = (self.eloss + self.rloss) / tf.cast(tf.shape(self.user_positive_items_pairs)[0], tf.float32)
        self.optimize = tf.train.AdamOptimizer(self.master_learning_rate).minimize(self.loss, var_list=[self.item_bias, self.item_embeddings, self.user_embeddings])

        # (N_USER_IDS, 1, K)
        user = tf.expand_dims(tf.nn.embedding_lookup(self.user_embeddings, self.score_user_ids), 1)
        # (1, N_ITEM, K)
        item = tf.expand_dims(self.item_embeddings, 0)
        self.item_scores = tf.reduce_sum(tf.multiply(user, item), 2) + tf.squeeze(self.item_bias)

        self.topk = tf.nn.top_k(self.item_scores, self.n_items)


def optimize(model, sampler, evaluator, num_batch, epoch, item_embedding_file):
    print ('Before initial tensorflow: {m}'.format(m = free_memory()))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    print ('After initial tensorflow: {m}'.format(m = free_memory()))

    _epoch = 0
    while _epoch < epoch:
        _losses, _users, _tops = [], evaluator.users(), []
        for _ in tqdm(range(num_batch), desc="Optimizing...", file=sys.stdout):
            user_pos, neg, flags = sampler.next_batch()
            #_, loss= sess.run((model.optimize, model.loss), {model.user_positive_items_pairs: user_pos, model.negative_samples: neg, model.negative_flags: flags})
            #options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            #run_metadata = tf.RunMetadata()
            _, impostor_num, impostor_log, loss, eloss, rloss = sess.run((
                model.optimize, model.impostor_num, model.impostor_log, model.loss, model.eloss, model.rloss),
                {model.user_positive_items_pairs: user_pos, model.negative_samples: neg, model.negative_flags: flags},
            #    options=options, run_metadata=run_metadata
            )
            #fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            #chrome_trace = fetched_timeline.generate_chrome_trace_format()
            #with open(os.path.join(output_path, 'timeline.json'), 'w') as f:
            #    f.write(chrome_trace)
            print (impostor_num, impostor_log, loss, eloss, rloss)
            _losses.append(loss)
        for chunk in chunks(_users, 100):
            _, _top = sess.run(model.topk, {model.score_user_ids: chunk})
            _tops.extend(_top)
        print("Training loss {}".format(np.mean(_losses)))
        print("Precision of validation set: {}".format(evaluator.eval(zip(_users, _tops), 50)))
        _epoch += 1
    item_embedding, item_bias = sess.run((model.item_embeddings, model.item_bias))
    with open(item_embedding_file, 'w') as f:
        for index, vec in enumerate(item_embedding.tolist()):
            f.write('%s\t%s\n' % (index, '|'.join(map(lambda x:'%.4f' % x, vec))))
    #with open(os.path.join(output_path, 'item_bias.txt'), 'w') as f:
    #    for index, vec in enumerate(item_bias.tolist()):
    #        f.write('%s\t%s\n' % (index, '|'.join(map(lambda x:'%.4f' % x, vec))))
    sampler.close()


if __name__ == '__main__':
    user_file, item_file, pos_pair_path, test_triplet_path, embed_dim, alpha, learning_rate, negative_num, batch_size, epoch, n_workers, item_embedding_file = sys.argv[1:]
    user2weight = {int(line.split('\t')[0]): 1 for line in open(user_file, 'r')}
    item2weight = {int(line.split('\t')[0]): np.log(int(line.split('\t')[2])) for line in open(item_file, 'r')}
    # Train a user-item joint embedding, where the items a user likes will be pulled closer to this users.
    # Once the embedding is trained, the recommendations are made by finding the k-Nearest-Neighbor to each user.
    total_sample = sum([int(line.split('\t')[2]) for line in open(item_file, 'r')])
    test_sample = sum([1 for fn in glob.glob(os.path.join(test_triplet_path, '*')) for line in open(fn, 'r')])
    num_batch = (total_sample - test_sample) / int(batch_size)
    model = WarpMF(n_users=len(user2weight),
                n_items=len(item2weight),
                # size of embedding
                embed_dim=int(embed_dim),
                # the size of hinge loss margin.
                margin=1,
                # clip the embedding so that their norm <= clip_norm
                alpha=float(alpha),
                # learning rate for AdaGrad
                master_learning_rate=float(learning_rate),
                )
    sampler = WarpSampler(pos_pair_path, batch_size=int(batch_size), n_workers=int(n_workers), item2weight=item2weight, negative_num=int(negative_num), start=0, end=len(user2weight) - 1)
    evaluator = Evaluator(test_triplet_path, 0, len(user2weight) - 1)
    optimize(model, sampler, evaluator, num_batch, int(epoch), item_embedding_file)
