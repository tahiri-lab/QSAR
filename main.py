import importlib
import json
import os

import pandas as pd
import yaml
import argparse

from qsar.utils.cross_validator import CrossValidator
from qsar.utils.extractor import Extractor
from qsar.preprocessing.custom_preprocessing import PreprocessingPipeline
from qsar.utils.hyperparameter_optimizer import HyperParameterOptimizer
from qsar.utils.visualizer import Visualizer

MODELS_PATH = "qsar.models."
CONFIG_PATH = "config/"


def save_results(model_name, R2, CV, custom_cv, Q2, output_dir):
    """
    Saves the model evaluation results in a JSON file.

    This function takes the performance metrics of a machine learning model and saves them in a JSON file within the specified output directory. It ensures the creation of the output directory if it does not already exist.

    :param model_name: Name of the model.
    :type model_name: str
    :param R2: R squared score of the model.
    :type R2: float
    :param CV: Cross-validation score of the model.
    :type CV: float
    :param custom_cv: Custom cross-validation score of the model.
    :type custom_cv: float
    :param Q2: Q squared score of the model.
    :type Q2: float
    :param output_dir: Path to the directory where results will be saved.
    :type output_dir: str
    :return: None
    """
    results = {
        'R2': R2,
        'CV': CV,
        'custom_cv': custom_cv,
        'Q2': Q2
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'{model_name}_results.json'), 'w') as f:
        json.dump(results, f)


def get_class_from_path(module_path: str, class_name: str) -> type:
    """
    Retrieves the class from the specified module and class name.

    :param module_path: Path to the module where the class is defined.
    :type module_path: str
    :param class_name: Name of the class to be retrieved.
    :type class_name: str
    :return: The class object specified by class_name from the module at module_path.
    :rtype: type
    """
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def main():
    """
    Main function orchestrating the machine learning pipeline for QSAR modeling.

    This function handles the following steps:

    - Parsing command-line arguments for configuration and output directory.
    - Loading configuration from a YAML file.
    - Initializing data extraction and preprocessing.
    - Setting up and executing cross-validation.
    - Dynamically loading, optimizing, and evaluating multiple machine learning models.
    - Visualizing model performance and predictions.

    The function reads command-line arguments for the configuration file and output directory,
    loads the configuration, prepares the data, performs cross-validation, optimizes and evaluates models,
    and finally visualizes the results.

    No parameters are required as input; all configurations are managed via command-line arguments and external files.

    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="compare_all_models.yaml", help="Path to the config file")
    parser.add_argument("--output", type=str, default="results", help="Path to the output directory")
    args = parser.parse_args()
    with open(CONFIG_PATH + args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

        # load the data
        extractor = Extractor(config["datasets"])

        df_full = pd.concat([extractor.get_df("full_train"), extractor.get_df("full_test")])

        # preprocessing pipeline
        preprocessing = PreprocessingPipeline(target=config["target"], variance_threshold=0, cols_to_ignore=[],
                                              verbose=False, threshold=0.9)
        pipeline = preprocessing.get_pipeline()
        pipeline.fit_transform(df_full)

        x_dfs, y_dfs = extractor.split_x_y(config["target"])
        df_full_tain = extractor.get_df("full_train")

        # cross validation
        cross_validator = CrossValidator(df_full_tain)
        visualizer = Visualizer()
        X_list, y_list, df, y, n_folds = cross_validator.create_cv_folds()
        visualizer.display_cv_folds(df, y, n_folds)

        # get the models
        models = dict()
        for model in config["models"]:
            model_name = str(model["name"])
            module_path = MODELS_PATH + model_name
            model_class_name = "".join(word.capitalize() for word in model_name.split("_")) + "Model"
            models[model_name] = get_class_from_path(module_path, model_class_name)

        # loop over the models
        # todo: parallelize this
        for model_name, model_class in models.items():
            model = model_class()
            optimizer = HyperParameterOptimizer(model=model, data=df_full_tain, direction='maximize', trials=100)
            study = optimizer.optimize()
            trial = study.best_trial
            print(f"Best trial: score {trial.value}, params {trial.params}")
            model.set_hyperparameters(**study.best_params)
            R2, CV, custom_cv, Q2 = cross_validator.evaluate_model_performance(
                model.model, x_dfs["full_train"], y_dfs["full_train"], x_dfs["full_test"], y_dfs["full_test"])
            visualizer.display_model_performance(model_name, R2, CV, custom_cv, Q2)
            y_full_train_pred, y_full_test_pred = cross_validator.get_predictions(model.model,
                                                                                  x_train=x_dfs["full_train"],
                                                                                  y_train=y_dfs["full_train"],
                                                                                  x_test=x_dfs["full_test"])
            visualizer.display_true_vs_predicted(model_name, y_dfs["full_train"], y_dfs["full_test"], y_full_train_pred,
                                                 y_full_test_pred)

            save_results(model_name, R2, CV, custom_cv, Q2, args.output)


if __name__ == "__main__":
    main()
