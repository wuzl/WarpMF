import os
import sys
import tensorflow as tf
from tqdm import tqdm
from time import sleep, time
from sampler import WarpSampler
from evaluator import Evaluator
from utils import chunks, free_memory
import numpy as np
from tensorflow.python.client import timeline
import glob


def parameter_server(cluster, n_items, embed_dim, ps_n, epoch, item_embedding_file):
    per_item_embedding = int(np.ceil(embed_dim * 1.0 / len(cluster.job_tasks('ps'))))
    per_item_bias = int(np.ceil(n_items * 1.0 / len(cluster.job_tasks('ps'))))
    item_embedding_chunk = list(chunks(range(embed_dim), per_item_embedding))
    item_bias_chunk = list(chunks(range(n_items), per_item_bias))
    with tf.device("/job:ps/task:%s/cpu:0" % ps_n):
        item_embeddings = tf.Variable(tf.random_normal([n_items, item_embedding_chunk[ps_n][-1] - item_embedding_chunk[ps_n][0] + 1], stddev=1 / (embed_dim ** 0.5), dtype=tf.float32), name="ps_item_embeddings_%s" % ps_n)
        item_bias =  tf.Variable(tf.random_normal([item_bias_chunk[ps_n][-1] - item_bias_chunk[ps_n][0] + 1, 1], stddev=1 / (embed_dim ** 0.5), dtype=tf.float32), name="ps_item_bias_%s" % ps_n)

    with tf.device("/job:ps/task:0/cpu:0"):
        task_queue = tf.FIFOQueue(len(cluster.job_tasks('ps') + cluster.job_tasks('worker')),[tf.bool], shared_name='ps_task_queue')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    print("Parameter server: start server")
    server = tf.train.Server(cluster, job_name="ps", task_index=ps_n, config=config, protocol="grpc+verbs")
    print("Parameter server: start session")
    sess = tf.Session(target=server.target)
    
    print("Parameter server: initializing variables; free memory: %s" % free_memory())
    sess.run(tf.global_variables_initializer())
    sess.run(task_queue.enqueue(True))
    while sess.run(task_queue.size())!=len(cluster.job_tasks('ps') + cluster.job_tasks('worker')):
        print("Parameter server: waiting...")
        sleep(5)
    print("Parameter server: variables initialized; free memory: %s" % free_memory())

    while sess.run(task_queue.size())!=len(cluster.job_tasks('ps')):
        sleep(5)

    _item_embeddings = sess.run(item_embeddings)
    with open(item_embedding_file % ps_n, 'w') as f:
        for index, vec in enumerate(_item_embeddings.tolist()):
            f.write('%s\t%s\n' % (index, '|'.join(map(lambda x:'%.4f' % x, vec))))
    print ("Parameter server done")
    # make sure all ps node save item embedding successfully
    sleep(10)


def worker(cluster, n_items, embed_dim, worker_n, epoch, alpha, master_learning_rate, sampler, evaluator, num_batch, num_user):
    per_item_embedding = int(np.ceil(embed_dim * 1.0 / len(cluster.job_tasks('ps'))))
    per_item_bias = int(np.ceil(n_items * 1.0 / len(cluster.job_tasks('ps'))))
    item_embedding_chunk = list(chunks(range(embed_dim), per_item_embedding))
    item_bias_chunk = list(chunks(range(n_items), per_item_bias))
    item_embedding_list = []
    item_bias_list = []
    for ps_n in range(len(cluster.job_tasks('ps'))):
        with tf.device("/job:ps/task:%s/cpu:0" % ps_n):
            item_embedding_list.append(tf.Variable(tf.random_normal([n_items, item_embedding_chunk[ps_n][-1] - item_embedding_chunk[ps_n][0] + 1], stddev=1 / (embed_dim ** 0.5), dtype=tf.float32), name="ps_item_embeddings_%s" % ps_n))
            item_bias_list.append(tf.Variable(tf.random_normal([item_bias_chunk[ps_n][-1] - item_bias_chunk[ps_n][0] + 1, 1], stddev=1 / (embed_dim ** 0.5), dtype=tf.float32), name="ps_item_bias_%s" % ps_n))

    with tf.device("/job:ps/task:0/cpu:0"):
        task_queue = tf.FIFOQueue(len(cluster.job_tasks('ps') + cluster.job_tasks('worker')),[tf.bool], shared_name='ps_task_queue')

    with tf.device("/job:worker/task:%s" % worker_n):
        item_embedding_copy_list = [tf.identity(item) for item in item_embedding_list]
        item_bias_copy_list = [tf.identity(item) for item in item_bias_list]
        item_embeddings_copy = tf.concat(item_embedding_copy_list, 1)
        item_bias_copy = tf.concat(item_bias_copy_list, 0)
        user_positive_items_pairs = tf.placeholder(tf.int32, [None, 2])
        negative_samples = tf.placeholder(tf.int32, [None, None])
        negative_flags = tf.placeholder(tf.float32, [None, None])
        score_user_ids = tf.placeholder(tf.int32, [None])
        user_embeddings = tf.Variable(tf.random_normal([num_user, embed_dim], stddev=1 / (embed_dim ** 0.5), dtype=tf.float32), name="worker_%s_user_embeddings" % worker_n)
        # N = batch size,
        # K = embedding size,
        # W = number of negative samples per a user-positive-item pair

        # user embedding (N, K)
        users = tf.nn.embedding_lookup(user_embeddings, user_positive_items_pairs[:, 0], name="worker_%s_users" % worker_n)
        user_reg = tf.reduce_sum(tf.square(users), 1, name="worker_%s_user_reg" % worker_n)
        # positive item embedding (N, K)
        pos_items = tf.nn.embedding_lookup(item_embeddings_copy, user_positive_items_pairs[:, 1])
        pos_reg = tf.reduce_sum(tf.square(pos_items), 1)
        pos_bias = tf.squeeze(tf.nn.embedding_lookup(item_bias_copy, user_positive_items_pairs[:, 1]))
        # positive item to user distance (N)
        pos_distances = tf.reduce_sum(tf.multiply(users, pos_items), 1 ) + pos_bias

        # negative item embedding (N, K, W)
        neg_items = tf.transpose(tf.nn.embedding_lookup(item_embeddings_copy, negative_samples), (0, 2, 1))
        neg_reg = tf.reduce_sum(tf.square(neg_items), 1)
        neg_bias = tf.squeeze(tf.nn.embedding_lookup(item_bias_copy, negative_samples))
        # distance to negative items (N x W)
        distance_to_neg_items = tf.reduce_sum(tf.multiply(tf.expand_dims(users, -1), neg_items), 1) + neg_bias

        impostors = tf.multiply(negative_flags, (tf.expand_dims(-pos_distances, -1) + distance_to_neg_items + 1))
        indexes = tf.where(tf.greater(impostors, 0))

        impostor_num = tf.shape(indexes)[0]
        impostor_log = tf.nn.moments(impostors, axes=[0, 1])

        x_min_y = tf.segment_min(indexes[:, 1], indexes[:, 0])
        uni_x, _ = tf.unique(indexes[:,0])
        uni_y = tf.nn.embedding_lookup(x_min_y, uni_x)
        xy = tf.concat([tf.expand_dims(uni_x, -1), tf.expand_dims(uni_y, -1)], 1)

        impostor_xy = tf.gather_nd(impostors, xy)
        rank = tf.log((n_items - 1) / tf.cast(uni_y + 1, tf.float32))

        eloss = tf.reduce_sum(tf.clip_by_value(rank * impostor_xy, 0, 10))
        rloss = tf.reduce_sum(alpha * (tf.gather_nd(neg_reg, xy) + tf.nn.embedding_lookup(pos_reg, uni_x) + tf.nn.embedding_lookup(user_reg, uni_x)))

        loss = (eloss + rloss) / tf.cast(tf.shape(user_positive_items_pairs)[0], tf.float32)
        optimizer = tf.train.AdamOptimizer(master_learning_rate)
        grads_and_vars = optimizer.compute_gradients(loss)
        dense_grads_and_vars = [(tf.convert_to_tensor(grad), var) for grad, var in grads_and_vars]
        opt = optimizer.apply_gradients(dense_grads_and_vars)
        item_scores = tf.reduce_sum(tf.multiply(tf.expand_dims(tf.nn.embedding_lookup(user_embeddings, score_user_ids), 1), tf.expand_dims(item_embeddings_copy, 0)), 2) + tf.squeeze(item_bias_copy)
        topk = tf.nn.top_k(item_scores, n_items)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    print("Worker %d: start server" %  worker_n)
    server = tf.train.Server(cluster, job_name="worker", task_index=worker_n, config=config, protocol="grpc+verbs")
    print("Worker %d: start session" %  worker_n)
    sess = tf.Session(target=server.target)
    
    print("Worker %d: initializing variables; free memory: %s" % (worker_n, free_memory()))
    sess.run(tf.global_variables_initializer())
    sess.run(task_queue.enqueue(True))
    while sess.run(task_queue.size())!=len(cluster.job_tasks('ps') + cluster.job_tasks('worker')):
        print("Worker %s: waiting..." % worker_n)
        sleep(1)
    print("Worker %d: variables initialized; free memory: %s" % (worker_n, free_memory()))

    _epoch = 0
    while _epoch < epoch:
        _losses, _users, _tops = [], evaluator.users(), []
        for _ in tqdm(range(num_batch), desc="Optimizing...", file=sys.stdout):
            user_pos, neg, flags = sampler.next_batch()
            #_, loss= sess.run((opt, loss), {user_positive_items_pairs: user_pos, negative_samples: neg, negative_flags: flags})
            #options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            #run_metadata = tf.RunMetadata()
            _, _impostor_num, _impostor_log, _loss, _eloss, _rloss = sess.run(
                [opt, impostor_num, impostor_log, loss, eloss, rloss],
                {user_positive_items_pairs: user_pos, negative_samples: neg, negative_flags: flags},
            #    options=options, run_metadata=run_metadata
            )
            print ("Worker %d: training" % worker_n, _impostor_num, _impostor_log, _loss, _eloss, _rloss)
            #fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            #chrome_trace = fetched_timeline.generate_chrome_trace_format()
            #with open(os.path.join(output_path, 'timeline_%s.json' % worker_n), 'w') as f:
            #    f.write(chrome_trace)
            #for device in run_metadata.step_stats.dev_stats:
            #    print(device.device)
            #    for node in device.node_stats:
            #        print("  ", node.node_name)
            _losses.append(_loss)
        print ("Worker %d: training" % worker_n, free_memory())
        for chunk in chunks(_users, 100):
            _, _top = sess.run(topk, {score_user_ids: chunk})
            _tops.extend(_top)
        print ("Worker %s: epoch: %s" % (worker_n, _epoch), free_memory(), np.mean(_losses), 50, evaluator.eval(zip(_users, _tops), 50), evaluator.eval(zip(_users, _tops), 50, 'recall'))
        _epoch += 1
    sess.run(task_queue.dequeue())
    sampler.close()
    print ("Worker %d: done" % worker_n)


if __name__ == '__main__':
    ps_config, worker_config, task, idx, user_file, item_file, pos_pair_path, test_triplet_path, embed_dim, alpha, learning_rate, negative_num, batch_size, epoch, n_workers, item_embedding_file = sys.argv[1:]
    cluster_config = {"worker": worker_config.split(','), "ps": ps_config.split(',')}
    user2weight = {int(line.split('\t')[0]): 1 for line in open(user_file, 'r')}
    item2weight = {int(line.split('\t')[0]): np.log(int(line.split('\t')[2])) for line in open(item_file, 'r')}
    total_sample = sum([int(line.split('\t')[2]) for line in open(item_file, 'r')])
    test_sample = sum([1 for fn in glob.glob(os.path.join(test_triplet_path, '*')) for line in open(fn, 'r')])
    cluster = tf.train.ClusterSpec(cluster_config)
    num_batch = (total_sample - test_sample) / (len(cluster_config['worker']) * int(batch_size))
    if task == 'ps':
        parameter_server(cluster, len(item2weight), int(embed_dim), int(idx), int(epoch), item_embedding_file)
    else:
        per_user = int(np.ceil(len(user2weight) * 1.0 / len(cluster_config['worker'])))
        user_chunk = list(chunks(range(len(user2weight)), per_user))
        start, end = user_chunk[int(idx)][0], user_chunk[int(idx)][-1]
        sampler = WarpSampler(pos_pair_path, batch_size=int(batch_size), n_workers=int(n_workers), item2weight=item2weight, negative_num=int(negative_num), start=start, end=end)
        evaluator = Evaluator(test_triplet_path, start, end)
        worker(cluster, len(item2weight), int(embed_dim), int(idx), int(epoch), float(alpha), float(learning_rate), sampler, evaluator, num_batch, end - start + 1)
