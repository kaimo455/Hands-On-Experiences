
import tensorflow as tf
import tensorflow_model_analysis as tfma
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils

import taxi_constants

_DENSE_FLOAT_FEATURE_KEYS = taxi_constants.DENSE_FLOAT_FEATURE_KEYS
_VOCAB_FEATURE_KEYS = taxi_constants.VOCAB_FEATURE_KEYS
_VOCAB_SIZE = taxi_constants.VOCAB_SIZE
_OOV_SIZE = taxi_constants.OOV_SIZE
_BUCKET_FEATURE_KEYS= taxi_constants.BUCKET_FEATURE_KEYS
_FEATURE_BUCKET_COUNT = taxi_constants.FEATURE_BUCKET_COUNT
_CATEGORICAL_FEATURE_KEYS = taxi_constants.CATEGORICAL_FEATURE_KEYS
_MAX_CATEGORICAL_FEATURE_VALUES = taxi_constants.MAX_CATEGORICAL_FEATURE_VALUES
_LABEL_KEY = taxi_constants.LABEL_KEY
_transformed_name = taxi_constants.transformed_name

def _transformed_names(keys):
    return [_transformed_name(key) for key in keys]


# Tf.Transform considers these features as "raw"
def _get_raw_feature_spec(schema):
    return schema_utils.schema_as_feature_spec(schema).feature_spec


def _gzip_reader_fn(filenames):
    """Small utility returning a record reader that can read gzip'ed files."""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def _build_estimator(config, hidden_units=None, warm_start_from=None):
    """Build an estimator for predicting the tipping behavior of taxi riders.
    Args:
        config: tf.estimator.RunConfig defining the runtime environment for the
          estimator (including model_dir).
        hidden_units: [int], the layer sizes of the DNN (input layer first)
        warm_start_from: Optional directory to warm start from.
    Returns:
        A dict of the following:
          - estimator: The estimator that will be used for training and eval.
          - train_spec: Spec for training.
          - eval_spec: Spec for eval.
          - eval_input_receiver_fn: Input function for eval.
    """
    real_valued_columns = [
        tf.feature_column.numeric_column(key, shape=())
        for key in _transformed_names(_DENSE_FLOAT_FEATURE_KEYS)
    ]
    categorical_columns = [
        tf.feature_column.categorical_column_with_identity(
            key,
            num_buckets=_VOCAB_SIZE+_OOV_SIZE,
            default_value=0
        )
        for key in _transformed_names(_VOCAB_FEATURE_KEYS)
    ]
    categorical_columns += [
        tf.feature_column.categorical_column_with_identity(
            key,
            num_buckets=_FEATURE_BUCKET_COUNT,
            default_value=0
        )
        for key in _transformed_names(_BUCKET_FEATURE_KEYS)
    ]
    categorical_columns += [
        tf.feature_column.categorical_column_with_identity(
            key,
            num_buckets=num_buckets,
            default_value=0
        )
        for key, num_buckets in zip(
            _transformed_names(_CATEGORICAL_FEATURE_KEYS),
            _MAX_CATEGORICAL_FEATURE_VALUES
        )
    ]
    return tf.estimator.DNNLinearCombinedClassifier(
        config=config,
        linear_feature_columns=categorical_columns,
        dnn_feature_columns=real_valued_columns,
        dnn_hidden_units=hidden_units or [100, 70, 50, 25],
        warm_start_from=warm_start_from
    )

def _example_serving_receiver_fn(tf_transform_graph, schema):
    """Build the serving in inputs.
    Args:
        tf_transform_graph: A TFTransformOutput.
        schema: The schema of the input data.
    Returns:
        Tensorflow graph which parses examples, applying tf-transform to them.
    """
    raw_feature_spec = _get_raw_feature_spec(schema)
    raw_feature_spec.pop(_LABEL_KEY)
    
    raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(raw_feature_spec, default_batch_size=None)
    serving_input_receiver = raw_input_fn()
    
    transformed_features = tf_transform_graph.transform_raw_features(serving_input_receiver.features)
    
    return tf.estimator.export.ServingInputReceiver(transformed_features, serving_input_receiver.receiver_tensors)

def _eval_input_receiver_fn(tf_transform_graph, schema):
    """Build everything needed for the tf-model-analysis to run the model.
    Args:
        tf_transform_graph: A TFTransformOutput.
        schema: The schema of the input data.
    Returns:
        EvalInputReceiver function, which contains:
          - Tensorflow graph which parses raw untransformed features, applies the 
            tf-transform preprocessing operators.
          - Set of raw, untransformed features.
          - Label against which predictions will be compared.
    """
    # Notice that the inputs are raw features, not transformed features here.
    raw_feature_spec = _get_raw_feature_spec(schema)
    
    serialized_tf_example = tf.compat.v1.placeholder(
        dtype=tf.string,
        shape=[None,],
        name='input_example_tensor'
    )
    
    # Add a parse_example operator to the tensorflow graph, which will parse
    # raw, untransformed, tf examples.
    features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
    
    # Now that we have our raw examples, process them through the tf-transform
    # function computed during the preprocessing step.
    transformed_features = tf_transform_graph.transform_raw_features(features)
    
    # The key name MUST be 'examples'
    receiver_tensors = {'examples': serialized_tf_example}
    
    # NOTE: Model is driven by transformed features (since training works on the
    # materialized output of TFT, but slicing will happen on raw features)
    features.update(transformed_features)
    
    return tfma.export.EvalInputReceiver(
        features=features,
        receiver_tensors=receiver_tensors,
        labels=transformed_features[_transformed_name(_LABEL_KEY)]
    )

def _input_fn(filenames, tf_transform_graph, batch_size=200):
    """Generate features and labels for training or evaluation.
    Args:
        filenames: [str] list of CSV files to read data from.
        tf_transform_graph, A TFTransformOutput.
        batch_size: int First dimension size of the Tensors returned by input_fn.
    Returns:
        A (features, indices) tuple where features is a dictionary of Tensors,
          and indices is a single Tensor of label indices.
    """
    transformed_feature_spec = tf_transform_graph.transformed_feature_spec().copy()
    
    dataset = tf.data.experimental.make_batched_features_dataset(
        filenames,
        batch_size,
        transformed_feature_spec,
        reader=_gzip_reader_fn
    )
    
    transformed_features = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
    # We pop the label because we do not want to use it as a feature while we're training
    return transformed_features, transformed_features.pop(_transformed_name(_LABEL_KEY))

# TFX will call this function
def trainer_fn(trainer_fn_args, schema):
    """Build the estimator using the high level API.
    Args:
        trainer_fn_args: Holds args used to train the model as name/value pairs.
        schema: Holds the schema of the training examples.
    Returns:
        A dict of the following:
          - estimator: The estimator that will be used for traning and eval.
          - train_spec: Spec for training.
          - eval_spec: Spec for eval.
          - eval_input_receiver_fn: Input function for eval.
    """
    # Number of nodes in the first layer of the DNN
    first_dnn_layer_size = 100
    num_dnn_layers = 4
    dnn_decay_factor = 0.7
    
    train_batch_size = 40
    eval_batch_size = 40
    
    tf_transform_graph = tft.TFTransformOutput(trainer_fn_args.transform_output)
    
    train_input_fn = lambda: _input_fn(
        filenames=trainer_fn_args.train_files,
        tf_transform_graph=tf_transform_graph,
        batch_size=train_batch_size
    )
    
    eval_input_fn = lambda: _input_fn(
        filenames=trainer_fn_args.eval_files,
        tf_transform_graph=tf_transform_graph,
        batch_size=eval_batch_size
    )
    
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=trainer_fn_args.train_steps
    )
    
    serving_receiver_fn = lambda: _example_serving_receiver_fn(tf_transform_graph, schema)
    
    exporter = tf.estimator.FinalExporter('chicago-taxi', serving_receiver_fn)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        steps=trainer_fn_args.eval_steps,
        exporters=[exporter],
        name='chicago-taxi-eval'
    )
    
    run_config = tf.estimator.RunConfig(save_checkpoints_steps=999, keep_checkpoint_max=1)
    run_config = run_config.replace(model_dir=trainer_fn_args.serving_model_dir)
    
    estimator = _build_estimator(
        config=run_config,
        hidden_units=[max(2, int(first_dnn_layer_size * dnn_decay_factor**i)) for i in range(num_dnn_layers)],
        warm_start_from=trainer_fn_args.base_model
    )
    
    # Create an input receiver for TFMA processing
    receiver_fn = lambda: _eval_input_receiver_fn(tf_transform_graph, schema)
    
    return {
        'estimator': estimator,
        'train_spec': train_spec,
        'eval_spec': eval_spec,
        'eval_input_receiver_fn': receiver_fn
    }