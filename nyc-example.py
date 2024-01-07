import ray

from ray.train import ScalingConfig, RunConfig
from ray.train.xgboost import XGBoostTrainer
from ray.train.xgboost import XGBoostPredictor

from ray import tune
from ray.tune import Tuner, TuneConfig

from ray import serve
from starlette.requests import Request

import requests, json
import numpy as np

ray.init()

dataset = ray.data.read_parquet("s3://anonymous@anyscale-training-data/intro-to-ray-air/nyc_taxi_2021.parquet").repartition(16)

train_dataset, valid_dataset = dataset.train_test_split(test_size=0.3)

trainer = XGBoostTrainer(
    label_column="is_big_tip",
    scaling_config=ScalingConfig(num_workers=4, use_gpu=False),
    params={ "objective": "binary:logistic", },
    datasets={"train": train_dataset, "valid": valid_dataset},
    run_config=RunConfig(storage_path='/mnt/cluster_storage/')
)

result = trainer.fit()

tuner = Tuner(trainer,
            param_space={'params' : {'max_depth': tune.randint(2, 12)}},
            tune_config=TuneConfig(num_samples=3, metric='train-logloss', mode='min'),
            run_config=RunConfig(storage_path='/mnt/cluster_storage/'))

checkpoint = tuner.fit().get_best_result().checkpoint

class OfflinePredictor:
    def __init__(self):
        import xgboost
        self._model = xgboost.Booster()
        self._model.load_model(checkpoint.path + '/model.json')

    def __call__(self, batch):
        import xgboost
        import pandas as pd
        dmatrix = xgboost.DMatrix(pd.DataFrame(batch))
        outputs = self._model.predict(dmatrix)
        return {"prediction": outputs}

predicted_probabilities = valid_dataset.drop_columns(['is_big_tip']).map_batches(OfflinePredictor, compute=ray.data.ActorPoolStrategy(size=2))

@serve.deployment
class OnlinePredictor:
    def __init__(self, checkpoint):
        import xgboost
        self._model = xgboost.Booster()
        self._model.load_model(checkpoint.path + '/model.json')

    async def __call__(self, request: Request) -> dict:
        data = await request.json()
        data = json.loads(data)
        return {"prediction": self.get_response(data) }

    def get_response(self, data):
        import pandas as pd
        import xgboost
        dmatrix = xgboost.DMatrix(pd.DataFrame(data, index=[0]))
        return self._model.predict(dmatrix)

handle = serve.run(OnlinePredictor.bind(checkpoint=checkpoint))

sample_input = valid_dataset.take(1)[0]
del(sample_input['is_big_tip'])
del(sample_input['__index_level_0__'])

requests.post("http://localhost:8000/", json=json.dumps(sample_input)).json()

serve.shutdown()
