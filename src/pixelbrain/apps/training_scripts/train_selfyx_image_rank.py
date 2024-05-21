# TODO: omerh -> remove this file once we open a repo for models
from pixelbrain.database_processors.xgboost_processor import XGBoostRegressorTrainer
from pixelbrain.database import Database
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb

mongo_key = os.getenv("MONGO_URL")


def plot_test_preds(test_preds, save_path):
    predictions = [item["prediction"] for item in test_preds]
    targets = [item["target"] for item in test_preds]

    # Create a scatter plot of predictions against targets
    plt.scatter(targets, predictions, alpha=0.5, edgecolors="k")
    plt.title("Test Predictions vs Targets")
    plt.xlabel("Target")
    plt.ylabel("Prediction")
    plt.plot(
        [min(targets), max(targets)], [min(targets), max(targets)], "r--"
    )  # Line y=x for reference
    plt.savefig(save_path)


def train_selfyx_image_rank():
    db = Database(database_id="Selfyx", mongo_key=mongo_key)

    data_field_names = [
        "cfg_scale",
        "pick_score",
        "similarity_score_nearest",
        "similarity_score_average_k_nearest",
        "similarity_score_maximum_distance",
        "generated_epoch",
    ]

    metric_field_name = "human_rating"
    filters = {field_name: None for field_name in data_field_names}
    filters[metric_field_name] = {"$gt": 0}
    data = db.find_images_with_filters(filters)
    data_df = pd.DataFrame(data)

    # add a column to the dataframe that is the log of the metric
    log_metric_field_name = f"log_{metric_field_name}"
    data_df[log_metric_field_name] = np.log(data_df[metric_field_name])

    # we train the model with a weighted MSE since the data is skewed towards lower ratings
    HIGH_SCORE_WEIGHTS = 1000

    def weighted_mse(y_true):
        weights = np.where(y_true >= np.log(4), HIGH_SCORE_WEIGHTS, 1)
        return weights

    from datetime import datetime

    current_date = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = f"selfyx-image-ranker-{current_date}"

    wandb.init(
        project="selfyx-image-ranker",
        name=run_name,
        config={
            "number_of_images": len(data_df),
            "features": data_field_names,
            "metric": log_metric_field_name,
            "high_score_weights": HIGH_SCORE_WEIGHTS,
        },
    )
    model_save_path = "xgboost_rating_model_log_weighted.pkl"
    log_trainer_weighted_metric_with_cross = XGBoostRegressorTrainer(
        data_df, data_field_names, log_metric_field_name, mse_weights_func=weighted_mse
    )
    log_weighted_test_preds = log_trainer_weighted_metric_with_cross.fit(
        save_model_path=model_save_path, auc_threshold=np.log(4)
    )
    wandb.log({"test_preds": log_weighted_test_preds})

    # log test predictions plot
    plt_save_path = "test_preds.png"
    plot_test_preds(log_weighted_test_preds, plt_save_path)
    wandb.log({"test_preds_plot": wandb.Image(plt_save_path)})

    # log model artifact
    model_metadata = {
        "training_field_names": data_field_names,
    }
    model_artifact = wandb.Artifact(
        name=f"xgboost_rating_model-{run_name}",
        type="model",
        metadata=model_metadata,
    )
    model_artifact.add_file(model_save_path)
    wandb.log_artifact(model_artifact, aliases="latest")

    # log dataset artifact
    data_save_path = "xgboost_training_data.csv"
    data_df.to_csv(data_save_path, index=False)
    data_artifact = wandb.Artifact(name=f"training_data-{run_name}", type="dataset")
    data_artifact.add_file(data_save_path)
    wandb.log_artifact(data_artifact, aliases="latest")
    print("Finished training model!")


if __name__ == "__main__":
    train_selfyx_image_rank()
