import absl
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

import os
import pprint
import tempfile
import urllib

import absl
import tensorflow as tf
import tensorflow_model_analysis as tfma
tf.get_logger().propogate = False
pp = pprint.PrettyPrinter()

import tfx
from tfx.components import CsvExampleGen
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import ExampleValidator
from tfx.components import Transform
from tfx.components import Trainer
from tfx.components import ResolverNode
from tfx.components import Evaluator
from tfx.components import Pusher

from tfx.dsl.experimental import latest_blessed_model_resolver

from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext

from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.proto.evaluator_pb2 import SingleSlicingSpec

from tfx.utils.dsl_utils import external_input

from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing


## Set up logging
absl.logging.set_verbosity(absl.logging.INFO)

## Download data
_data_root = tempfile.mkdtemp(prefix='tfx-data')
DATA_PATH = 'https://raw.githubusercontent.com/tensorflow/tfx/master/tfx/examples/chicago_taxi_pipeline/data/simple/data.csv'
_data_filepath = os.path.join(_data_root, 'data.csv')
urllib.request.urlretrieve(DATA_PATH, _data_filepath)

## Create interacitveContext
context = InteractiveContext()

example_gen = CsvExampleGen(input=external_input(_data_root))
context.run(example_gen)

statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
context.run(statistics_gen)

schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'], infer_feature_shape=False)
context.run(schema_gen)

example_validator = ExampleValidator(statistics=statistics_gen.outputs['statistics'], schema=schema_gen.outputs['schema'])
context.run(example_validator)

_taxi_constants_module_file = 'taxi_constants.py'

_taxi_transform_module_file = 'taxi_transform.py'

transform = Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file=os.path.abspath(_taxi_transform_module_file)
)
context.run(transform)

_taxi_trainer_module_file = 'taxi_trainer.py'

trainer = Trainer(
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    module_file=os.path.abspath(_taxi_trainer_module_file),
    train_args=trainer_pb2.TrainArgs(num_steps=10000),
    eval_args=trainer_pb2.EvalArgs(num_steps=5000)
)
context.run(trainer)

eval_config = tfma.EvalConfig(
    model_specs=[
        # Using signature 'eval' implies the use of an EvalSavedModel. To use
        # a serving model remove the signature to defaults to 'serving_default'
        # and add a label_key.
        tfma.ModelSpec(signature_name='eval')
    ],
    metrics_specs=[
        tfma.MetricsSpec(
            # The metrics added here are in addition to those saved with the
            # model (assuming either a keras model or EvalSavedModel is used).
            # Any metrics added into the saved model (for example using
            # model.compile(..., metrics=[...]), etc) will be computed
            # automatically.
            metrics=[tfma.MetricConfig(class_name='ExampleCount')],
            # To add validation thresholds for metrics saved with the model,
            # add them keyed by metric name to the thresholds map.
            thresholds={
                'accuracy': tfma.MetricThreshold(
                    value_threshold=tfma.GenericValueThreshold(lower_bound={'value': 0.5}),
                    change_threshold=tfma.GenericChangeThreshold(
                        direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                        absolute={'value': -1e-10}
                    )
                )
            }
        )
    ],
    slicing_specs=[
        # An empty slice spec means the overall slice, i.e. the whole dataset.
        tfma.SlicingSpec(),
        # Data can be sliced along a feature column. In this case, data is
        # sliced along feature column trip_start_hour.
        tfma.SlicingSpec(feature_keys=['trip_start_hour'])
    ]
)

# Use TFMA to compute a evaluation statistics over features of a model and
# validate them against a baseline.

# The model resolver is only required if performing model validation in addition
# to evaluation. In this case we validate against the latest blessed model. If
# no model has been blessed before (as in this case) the evaluator will make our
# candidate the first blessed model.
model_resolver = ResolverNode(
    instance_name='latest_blessed_model_resolver',
    resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
    model=Channel(type=Model),
    model_blessing=Channel(type=ModelBlessing))
context.run(model_resolver)

evaluator = Evaluator(
    examples=example_gen.outputs['examples'],
    model=trainer.outputs['model'],
    #baseline_model=model_resolver.outputs['model'],
    # Change threshold will be ignored if there is no baseline (first run).
    eval_config=eval_config)
context.run(evaluator)

_serving_model_dir = os.path.join(tempfile.mkdtemp(), 'serving_model/taxi_simple')
pusher = Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    push_destination=pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(base_directory=_serving_model_dir)
    )
)
context.run(pusher)

_runner_type = 'beam'
_pipeline_name = 'chicage_taxi_{:s}'.format(_runner_type)

_tfx_root = os.path.join(os.environ['HOME'], 'tfx')
_taxi_root = os.path.join(os.environ['HOME'], 'taxi')
_serving_model_dir = os.path.join(_taxi_root, 'serving_model')
_data_root = os.path.join(_taxi_root, 'data', 'simple')
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)
_metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name,
                              'metadata.db')

components = [
    example_gen, statistics_gen, schema_gen, example_validator, transform,
    trainer, evaluator, pusher
]











absl.logging.set_verbosity(absl.logging.INFO)

tfx_pipeline = pipeline.Pipeline(
    pipeline_name=_pipeline_name,
    pipeline_root=_pipeline_root,
    components=components,
    enable_cache=True,
    metadata_connection_config=(
        metadata.sqlite_metadata_connection_config(_metadata_path)),

    # direct_num_workers=0 means auto-detect based on the number of CPUs
    # available during  execution time.
    #
    # TODO(b/142684737): The multi-processing API might change.
    beam_pipeline_args = ['--direct_num_workers=0'],

    additional_pipeline_args={})

BeamDagRunner().run(tfx_pipeline)