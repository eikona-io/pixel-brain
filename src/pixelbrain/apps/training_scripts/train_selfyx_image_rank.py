# TODO: omerh -> remove this file once we open a repo for models
from pixelbrain.database_processors.xgboost_processor import XGBoostRankerTrainer
from pixelbrain.database import Database
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb
from datetime import datetime

mongo_key = os.getenv("MONGO_URL")


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
    group_by_field_name = "session_id"
    filters = {field_name: None for field_name in data_field_names}
    filters[metric_field_name] = {"$gt": 0}
    data = db.find_images_with_filters(filters)
    data_df = pd.DataFrame(data)

    current_date = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = f"selfyx-image-ranker-{current_date}"

    wandb.init(
        project="selfyx-image-ranker",
        name=run_name,
        config={
            "number_of_images": len(data_df),
            "features": data_field_names,
            "metric": metric_field_name,
        },
    )
    model_save_path = "xgboost_rating_model.pkl"
    trainer = XGBoostRankerTrainer(
        data_df,
        data_field_names,
        metric_field_name,
        group_by_field_name=group_by_field_name,
    )
    test_ndcg, best_params = trainer.fit(save_model_path=model_save_path)

    print(f"Test NDCG: {test_ndcg}")
    print(f"Best params: {best_params}")
    wandb.log({"test_ndcg": test_ndcg, "best_params": best_params})

    # log model artifact
    model_metadata = {
        "training_field_names": data_field_names,
        "test_ndcg": test_ndcg,
        "best_params": best_params,
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
