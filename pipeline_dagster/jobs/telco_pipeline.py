from dagster import job
from pipeline_dagster.ops.data_prep_op import data_prep_op
from pipeline_dagster.ops.train_op import train_op

@job
def telco_pipeline():
    train_op(data_prep_op())

