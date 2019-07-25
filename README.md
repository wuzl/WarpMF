# WarpMF
A Tensorflow implementation of warp loss matrix factorization.

# Features
* Produces embedding that accurately captures the user-item, user-user, and item-item similarity. 
* Outperforms state-of-the-art recommendation algorithms on a wide range of tasks.
* Enjoys an extremely efficient Top-K search using Fast KNN algorithms.

# Utility Features
* Parallel negative sampler that can sample the user-item pairs when the model is being trained on GPU.
* Fast precision evaluation based on Tensorflow.

# Requirements
 * python
 * tensorflow
 * numpy
 * tqdm

# Usage
```bash
export LD_LIBRARY_PATH=/opt/cuda/extras/CUPTI/lib64/:.
# install requirements
pip install -r requirements.txt
# run demo tensorflow model
python train.py /home2/alg/hin/vector/tmp/UT.user /home2/alg/hin/vector/tmp/UT.item /home2/alg/hin/vector/tmp/UT.train.pair /home2/alg/hin/vector/tmp/UT.test.triplet 200 0.1 0.001 100 10000 20 3 /home2/alg/hin/vector/tmp/UT.item_embedding
python train.py /home2/alg/hin/vector/tmp/UM.user /home2/alg/hin/vector/tmp/UM.item /home2/alg/hin/vector/tmp/UM.train.pair /home2/alg/hin/vector/tmp/UM.test.triplet 200 0.05 0.0005 100 10000 20 3 /home2/alg/hin/vector/tmp/UM.item_embedding
python distributed_train.py alice1b:3335,alice2b:3336 alice1b:3333,alice2b:3334 ps/worker 0/1 /home2/alg/hin/vector/tmp/UT.user /home2/alg/hin/vector/tmp/UT.item /home2/alg/hin/vector/tmp/UT.train.pair /home2/alg/hin/vector/tmp/UT.test.triplet 200 0.1 0.001 100 10000 20 3 /home2/alg/hin/vector/tmp/UT.item_embedding
tfrun -w 2 -s 2 -Gw 1 -Gs 17 -Mw 10000 -Ms 2000 -m alice1a -- python -u distributed_train.py {ps_hosts} {worker_hosts} {job_name} {task_index} /home2/alg/hin/vector/tmp/UT.user /home2/alg/hin/vector/tmp/UT.item /home2/alg/hin/vector/tmp/UT.train.pair /home2/alg/hin/vector/tmp/UT.test.triplet 200 0.1 0.001 100 10000 20 3 /home2/alg/hin/vector/tmp/UT.item_embedding
```

# Acknowledgement
The code is based on [CollMetric](https://github.com/changun/CollMetric) and paper [Wsabie: Scaling Up To Large Vocabulary Image Annotation](http://www.thespermwhale.com/jaseweston/papers/wsabie-ijcai.pdf)

# TODO
* Model Comparison.
* TensorBoard visualization.
