"""train the model"""

import argparse
import configparser
import logging
import reprlib
import sys
import os

import tensorflow as tf

sys.path.append("../..")

from FCGen import FCGen

def input_fn(record_files, spec, shuffle=True, batch_size=64, epochs=1):
  """General input functions

  Args:
    record_files: (list) A list of tfrecord files
    spec: (dict) feature column parsing specification
    shuffle: (bool) whether to shuffle
    batch_size: (int) batch size

  Returns:
    dataset batch iterator and init op
  """
  files = tf.data.Dataset.from_tensor_slices(record_files)
  #dataset = tf.data.TFRecordDataset(record_files)
  dataset = files.interleave(tf.data.TFRecordDataset, 6)

  if epochs > 1:
    dataset = dataset.repeat(epochs)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=batch_size * 1000)

  def _map(example_proto):
    features = tf.parse_example(example_proto, spec)
    labels = tf.cast(features.pop('label'), tf.int64)
    return features, labels

  #dataset = dataset.apply(tf.contrib.data.map_and_batch(
  #    map_func=_map, batch_size=batch_size, num_parallel_batches=56))
  dataset = dataset.batch(batch_size)
  dataset = dataset.map(_map, num_parallel_calls=8)

  dataset = dataset.prefetch(1)

  iterator = dataset.make_one_shot_iterator()
  next_batch = iterator.get_next()

  #features = tf.parse_example(next_batch, features=spec)
  #labels = tf.cast(features.pop('label'), tf.int64)
  #features.pop('wt')

  features, labels = next_batch[0], next_batch[1]
  #print(features)
  #print(labels)

  return features, labels

def model_fn(features, labels, mode, params):
  units = params.get('units', 1)
  columns = params['columns']

  cols_to_vars = {}
  print(features)
  logits = tf.feature_column.linear_model(
                  features=features,
                  feature_columns=columns,
                  units=units,
                  cols_to_vars=cols_to_vars)

  prediction = tf.nn.sigmoid(logits)

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
      'logits': logits
    }

    return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  loss = tf.losses.sigmoid_cross_entropy(labels, logits=logits, reduction=tf.losses.Reduction.SUM)
  print(loss)

  loss = tf.losses.compute_weighted_loss(loss, reduction=tf.losses.Reduction.SUM)
  
  auc = tf.metrics.auc(labels=labels, predictions=prediction)

  metrics = {'auc': auc}
  tf.summary.scalar('auc', auc[0])
  
  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)

  assert mode == tf.estimator.ModeKeys.TRAIN

  optimizer = tf.train.FtrlOptimizer(
                    learning_rate=params['learning_rate'],
                    l1_regularization_strength=params['l1_reg'],
                    l2_regularization_strength=params['l2_reg'])
  train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
