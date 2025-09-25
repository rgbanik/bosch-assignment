import pandas as pd
import os
import ijson
import streamlit as st

class StatisticsRow:
    """
    This class is used to store the statistics for each class.
    It is also used to shorten code based on OOP principles.
    """
    def __init__(self):
        self.values = {
            "total_count": 0,
            "occluded_count": 0,
            "truncated_count": 0,
            "max_area": float('-inf'),
            "min_area": float('inf'),
            "sum_area": 0,
            "mean_area": 0,
            "max_width": float('-inf'),
            "min_width": float('inf'),
            "sum_width": 0,
            "mean_width": 0,
            "max_height": float('-inf'),
            "min_height": float('inf'),
            "sum_height": 0,
            "mean_height": 0

        }

    def __getitem__(self, key):
        return self.values[key]

    def __setitem__(self, key, value):
        self.values[key] = value

    def as_dict(self):
        return self.values


class DatasetAnalyzer:
    """
    This class is used to analzye a dataset split,
    and generate the necessary csv files and plots
    to be used in the report.
    """
    def __init__(self):
        self.labels_root = "/data/labels"
        self.image_width = 1280
        self.image_height = 720
        self.splits = [ "train", "val" ]
        self.categories = {
            "traffic sign",
            "traffic light",
            "car",
            "rider",
            "motor",
            "person",
            "bus",
            "truck",
            "bike",
            "train"
        }
        self.class_stats = {
            "traffic sign": StatisticsRow(),
            "traffic light": StatisticsRow(),
            "car": StatisticsRow(),
            "rider": StatisticsRow(),
            "motor": StatisticsRow(), # Motorcycle, haha
            "person": StatisticsRow(),
            "bus": StatisticsRow(),
            "truck": StatisticsRow(),
            "bike": StatisticsRow(),
            "train": StatisticsRow()
        }

    def __getitem__(self, key):
        return self.class_stats[key]

    def __setitem__(self, key, value):
        self.class_stats[key] = value

    def as_dict(self):
        return {
            key: value.as_dict()
            for key, value in self.class_stats.items()
        }

    def as_dataframe(self):
        return pd.DataFrame(self.as_dict())

    def save_df_as_csv(self, split):
        os.makedirs("/data/csv", exist_ok=True)
        st.write(f"This is what the {split} statistics looks like:")
        st.dataframe(self.as_dataframe())
        self.as_dataframe().to_csv(
            f"/data/csv/{split}_stats.csv"
        )
        print(f"Saved {split} stats to /data/csv/{split}_stats.csv")

    def reset_class_stats(self):
        self.class_stats = {
            "traffic sign": StatisticsRow(),
            "traffic light": StatisticsRow(),
            "car": StatisticsRow(),
            "rider": StatisticsRow(),
            "motor": StatisticsRow(),
            "person": StatisticsRow(),
            "bus": StatisticsRow(),
            "truck": StatisticsRow(),
            "bike": StatisticsRow(),
            "train": StatisticsRow()
        }

    def analyze_labels(self):
        """
        This method is for extracting counts from the json file,
        so that we can infer useful information and generate
        a csv file which can be used to render plots
        """
        st.write("We start by generating some statistics for our dataset splits.")
        for split in self.splits:
            self.reset_class_stats()
            labels_path = f"{self.labels_root}/bdd100k_labels_images_{split}.json"
            with open(labels_path, "r") as f:
                for obj in ijson.items(f, "item"):
                    for label in obj["labels"]:
                        if label["category"] in self.categories:
                            self[label["category"]]["total_count"] += 1
                            if label["attributes"]["occluded"]:
                                self[label["category"]]["occluded_count"] += 1
                            if label["attributes"]["truncated"]:
                                self[label["category"]]["truncated_count"] += 1
                            box = label["box2d"]
                            area = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
                            self[label["category"]]["max_area"] = max(self[label["category"]]["max_area"], area)
                            self[label["category"]]["min_area"] = min(self[label["category"]]["min_area"], area)
                            self[label["category"]]["sum_area"] += area
                            self[label["category"]]["max_width"] = max(self[label["category"]]["max_width"], box["x2"] - box["x1"])
                            self[label["category"]]["min_width"] = min(self[label["category"]]["min_width"], box["x2"] - box["x1"])
                            self[label["category"]]["sum_width"] += box["x2"] - box["x1"]
                            self[label["category"]]["max_height"] = max(self[label["category"]]["max_height"], box["y2"] - box["y1"])
                            self[label["category"]]["min_height"] = min(self[label["category"]]["min_height"], box["y2"] - box["y1"])
                            self[label["category"]]["sum_height"] += box["y2"] - box["y1"]
                # Now that the counts have been computed, we can compute the means
                for category in self.categories:
                    self[category]["mean_area"] = self[category]["sum_area"] / self[category]["total_count"]
                    self[category]["mean_width"] = self[category]["sum_width"] / self[category]["total_count"]
                    self[category]["mean_height"] = self[category]["sum_height"] / self[category]["total_count"]
                self.save_df_as_csv(split)
