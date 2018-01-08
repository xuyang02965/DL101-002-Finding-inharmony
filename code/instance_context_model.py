# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from itertools import compress
import time
import os

PAD_IDX = 0    # 用于填充的分类索引
UNK_IDX = 91   # 不在coco已知分类中的物体索引

VOCAB_SIZE = 92  # 所有分类的个数
EMBEDDING_SIZE = 256  
CONTEXT_SIZE = 16  # 每幅图片中物体类别数
#CONTEXT_SIZE = 64  # 每幅图片中物体数

cocoCatIds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, \
              24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, \
              48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, \
              72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

mask = np.zeros(shape=(VOCAB_SIZE,), dtype=np.float32)
for catId in cocoCatIds:
    mask[catId] = 1.
mask[PAD_IDX] = 1.
mask[UNK_IDX] = 1.

def build_model_graph(learning_rate=1e-3):
    # 在训练时，输入的是单个物体类别，输出是与其共现的所有其它物体类别的出现概率
    # 在预测时，输入的是图片中所有物体，输出是与当前物体共现的所有其它物体类别的出现概率
    inputs_ = tf.placeholder(tf.int32, shape=[None, CONTEXT_SIZE])
    labels_ = tf.placeholder(tf.float32, shape=[None, VOCAB_SIZE])
    
    # embedding层
    embeddings = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBEDDING_SIZE]))
    input_embeddings = tf.nn.embedding_lookup(embeddings, inputs_)
    context_embeddings = tf.reduce_sum(input_embeddings, axis=1)
    
    # hidden层
    hidden_output = tf.layers.dense(context_embeddings, EMBEDDING_SIZE, activation=tf.nn.tanh)
    
    # output层
    raw_output = tf.layers.dense(hidden_output, VOCAB_SIZE)
    final_output = tf.sigmoid(raw_output)
    
    # loss and GD optimizer
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=raw_output, labels=labels_)*mask
        # tf.nn.sparse_softmax_cross_entropy_with_logits(logits=raw_output, labels=labels_) * mask
    )
    
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    
    saver = tf.train.Saver()
    
    init = tf.global_variables_initializer()
    
    return init, inputs_, labels_, final_output, loss, train_step, saver


def model_train(session, inputs_, labels_, final_output, loss, train_step):
    batch_size = 256
    epochs = 1000
    average_loss = 0
    step = 0
    eval_interval = 1000
    start_time = time.time()
    try:
        for epoch in range(epochs):
            num_inputs = len(train_data)
            order = np.arange(num_inputs)
            np.random.shuffle(order)
            for j in range(0, num_inputs, batch_size):
                step += 1
                batch_index = order[j: j + batch_size]
                batch_inputs = train_data[batch_index]
                batch_labels = train_labels[batch_index]
                feed_dict = {inputs_: batch_inputs, \
                             labels_: batch_labels}
                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                _, loss_val = session.run([train_step, loss], feed_dict=feed_dict)
                average_loss += loss_val
                # if step == 100:
                #     break
    
                if step % eval_interval == 0:
                    average_loss /= eval_interval
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print('Average loss at step ', step, ': ', average_loss)
                    average_loss = 0
    
    except KeyboardInterrupt:
        print('Interrupted')
    finally:
        end_time = time.time()
        print('\ntime: {:.2f} s'.format(end_time - start_time))

def model_save(session, saver, model_base_path = './models/', model_name = 'instance-context-model'):
    saver.save(session, os.path.join(model_base_path, model_name))

def model_restore(session, saver, model_base_path = './models/', model_name = 'instance-context-model'):
    saver.restore(session, os.path.join(model_base_path, model_name))
    
def model_inference(session, inputs, inputs_ph, final_output):
    feed_dict = {inputs_ph: inputs}
    output_val = session.run([final_output], feed_dict=feed_dict)
    
    results_list = list()
    for res in output_val:
        for item in res:
            results_list.append(item)
    
    r = list(results_list[0]*mask)
    return r

def get_coco_catIds(classIds):
    catIds = list()
    for classId in classIds:
        catIds.append(cocoCatIds[classId - 1])
    return catIds

def validate_batch(session, batch_inputs, inputs_ph, final_output):
    rets = list()
    for single_input in batch_inputs:
        ret = validate_one(session, list(single_input), inputs_ph, final_output)
        rets.append(ret)
    return rets

def validate_one(session, ins_cat_ids, inputs_ph, final_output):
    all_catIds = set(ins_cat_ids)
    catIds = []
    inputs = np.zeros(shape=(1, CONTEXT_SIZE), dtype=np.int32)
    for i in range(len(ins_cat_ids)):
        inputs[0][i] = ins_cat_ids[i]
    
    while len(catIds) != len(all_catIds):
        r = model_inference(session, inputs, inputs_ph, final_output)
        candidates = dict()
        for catId in all_catIds:
            if not (catId in catIds):
                candidates[catId] = r[catId]
                # print (catId, r[catId])
        newcid = max(candidates, key=candidates.get)
        # print (newcid)
        # print ("*"*30)
        catIds += [newcid]
        inputs = np.zeros(shape=(1, CONTEXT_SIZE), dtype=np.int32)
        for i in range(len(ins_cat_ids)):
            if ins_cat_ids[i] in catIds:
                inputs[0][i] = ins_cat_ids[i]
    
    # print (catIds)
    return catIds


def train():
    tf.reset_default_graph()
    learning_rate = 1e-3
    init, inputs_, labels_, \
    final_output, loss, \
    train_step, saver = build_model_graph(learning_rate)

    with tf.Session() as session:
        session.run(init)
        print('Initialized')
        model_train(session, inputs_, labels_, final_output, loss, train_step)
        model_save(session, saver)

# 将数据从输入为单个物体类别，转换为图片包含的所有物体类别，实际就是增加了PADDING。将原来的N*1维数组，变成了N*CONTEXT_SIZE维数组
def prepare_data(raw_data):
    data = np.zeros(shape=(len(raw_data), CONTEXT_SIZE), dtype=np.int32)
    for i in range(len(raw_data)):
        for j in range(len(raw_data[i])):
            if j == CONTEXT_SIZE:
                break;
            data[i][j] = raw_data[i][j]
    return data

def prepare_labels(raw_labels):
    labels = np.zeros(shape=(len(raw_labels), VOCAB_SIZE), dtype=np.float32)
    for i in range(len(raw_labels)):
        for item in raw_labels[i]:
            labels[i][item] = 1.
    return labels



#train_rdata = np.load("data/train_data.npy")
#train_data = prepare_data(train_rdata)
#train_rlabels = np.load("data/train_labels.npy")
#train_labels = prepare_labels(train_rlabels)

# train()
