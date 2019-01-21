import tensorflow as tf
from FCGen.Generator import *

def GetFeatureSpec(config):
  for sec, info_dict in [(sec, config[sec]) for sec in config.sections()]:
    col_type = config[sec]["ftype"]

    if col_type == "numeric":
      GetNumericColumn(sec, info_dict)
    elif col_type == "bucketized":
      GetBucketizedColumn(sec, info_dict)
    elif col_type == "cat_id":
      GetCatIdentityColumn(sec, info_dict)
    elif col_type == "cat_hash":
      GetCatHashColumn(sec, info_dict)
    elif col_type == "cat_vocab":
      GetCatVocabColumn(sec, info_dict)
    elif col_type == "cross":
      GetCrossColumn(sec, info_dict)
    elif col_type == "embedding":
      GetEmbeddingColumn(sec, info_dict)
    elif col_type == "shared_embedding":
      GetSharedEmbeddingColumn(sec, info_dict)
    else:
      assert False, "Unsupported column type: %s" % (col_type)

  all_columns = out_columns.items()

  #for name, c in all_columns:
    ##print("parse: " + name)
    ##print(c)
  #  tmp_spec = tf.feature_column.make_parse_example_spec([c])

  feature_spec = tf.feature_column.make_parse_example_spec(
     [c for _, c_lst in all_columns for c in c_lst])
  
  return {g : [c for k, c_lst in lst.items() for c in c_lst if not k in ('label')] for g, lst in group_columns.items()}, \
         feature_spec

