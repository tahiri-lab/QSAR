import os
import yaml
import argparse

import neptune.new as neptune

from qsar.utils.extractor import Extractor
from qsar.preprocessing.custom_preprocessing import PreprocessingPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="compare_all_models.yaml.yaml", help="Path to the config file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

        # todo: add in the docs how to set up the api key, setx for windows and export for linux
        api_key = os.environ["NEPTUNE_API_KEY"]
        run = neptune.init_run(project="moben1/QSAR", api_token=api_key)

        # load the data
        extractor = Extractor(config["datasets"])

        # preprocessing pipeline
        preprocessing = PreprocessingPipeline(target=config["target"], variance_threshold=0, cols_to_ignore=[],
                                              verbose=False, threshold=0.9)
        pipeline = preprocessing.get_pipeline()
        pipeline.fit_transform(extractor.get_df("train"))

        # model pipeline
        # todo: add the model pipeline and the model training

        # models comparison
        # todo: compare the models and log the results to neptune

        run.stop()


if __name__ == "__main__":
    main()
