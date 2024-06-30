"""Evaluation script for measuring K-means clustering quality."""
import json
import logging
import pathlib
import pickle
import tarfile

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.debug("Starting evaluation.")
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    logger.debug("Loading K-means model.")
    model = pickle.load(open("model.pkl", "rb"))

    logger.debug("Reading test data.")
    test_path = "/opt/ml/processing/test/test.csv"
    df = pd.read_csv(test_path, header=None)

    logger.info("Performing predictions against test data.")
    predictions = model.predict(df.values)

    logger.debug("Calculating silhouette score and inertia.")
    silhouette = silhouette_score(df.values, predictions)
    inertia = model.inertia_

    report_dict = {
        "clustering_metrics": {
            "silhouette_score": {
                "value": float(silhouette),
            },
            "inertia": {
                "value": float(inertia),
            },
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Writing out evaluation report with silhouette score: %f and inertia: %f", silhouette, inertia)
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))

    # Optional: Visualize clusters (if 2D or 3D)
    if df.shape[1] in [2, 3]:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        plt.figure(figsize=(10, 8))
        if df.shape[1] == 2:
            plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=predictions, cmap='viridis')
        else:  # 3D
            ax = plt.axes(projection='3d')
            ax.scatter3D(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], c=predictions, cmap='viridis')
        
        plt.title('K-means Clustering Results')
        plt.savefig(f"{output_dir}/clustering_visualization.png")
        plt.close()

    logger.info("Evaluation complete.")