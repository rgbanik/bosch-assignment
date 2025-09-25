import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import ijson
import pandas as pd
import streamlit as st

class Visualizer:
    """
    Used to generate visualizations from the csv files,
    and also compute histograms and violin plots on-the-fly.
    """
    def __init__(self):
        self.splits = ["train", "val"]
        self.box_sizes = {
            "traffic sign": [],
            "traffic light": [],
            "car": [],
            "rider": [],
            "motor": [],
            "person": [],
            "bus": [],
            "truck": [],
            "bike": [],
            "train": []
        }
        self.box_centers = {
            "traffic sign": [],
            "traffic light": [],
            "car": [],
            "rider": [],
            "motor": [],
            "person": [],
            "bus": [],
            "truck": [],
            "bike": [],
            "train": []
        }
        self.labels_root = "/data/labels"
        self.csv_root = "/data/csv"
        self.image_width = 1280
        self.image_height = 720
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

    def reset_box_sizes(self):
        self.box_sizes = {
            "traffic sign": [],
            "traffic light": [],
            "car": [],
            "rider": [],
            "motor": [],
            "person": [],
            "bus": [],
            "truck": [],
            "bike": [],
            "train": []
        }

    def reset_box_centers(self):
        self.box_centers = {
            "traffic sign": [],
            "traffic light": [],
            "car": [],
            "rider": [],
            "motor": [],
            "person": [],
            "bus": [],
            "truck": [],
            "bike": [],
            "train": []
        }

    def render_barplots_comparison(self):
        df_train = pd.read_csv("/data/csv/train_stats.csv", index_col=0).T
        df_val = pd.read_csv("/data/csv/val_stats.csv", index_col=0).T

        classes = df_train.index
        x = np.arange(len(classes))
        width = 0.35

        # Compute visible counts
        train_visible = df_train['total_count'] - df_train['occluded_count'] - df_train['truncated_count']
        val_visible = df_val['total_count'] - df_val['occluded_count'] - df_val['truncated_count']

        # Plot stacked bars for train
        fig, ax = plt.subplots(figsize=(12,6))
        ax.bar(x - width/2, train_visible, width, label='Visible (train)', color='skyblue')
        ax.bar(x - width/2, df_train['occluded_count'], width, bottom=train_visible, label='Occluded', color='orange')
        ax.bar(x - width/2, df_train['truncated_count'], width, bottom=train_visible + df_train['occluded_count'], label='Truncated', color='red')

        # Plot stacked bars for val
        ax.bar(x + width/2, val_visible, width, label='Visible (val)', color='lightgreen')
        ax.bar(x + width/2, df_val['occluded_count'], width, bottom=val_visible, label='_nolegend_', color='orange')
        ax.bar(x + width/2, df_val['truncated_count'], width, bottom=val_visible + df_val['occluded_count'], label='_nolegend_', color='red')

        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45)
        ax.set_ylabel("Count")
        ax.set_title("Stacked Counts: Visible, Occluded, Truncated (Train vs Validation)")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
        st.write("We see here that a huge number of cars in the train set are either truncated and occluded.")
        st.write("While this is undesirable in a small dataset, the large size of this dataset will help the model identify cars in various settings.")

        classes = df_train.index
        x = np.arange(len(classes))  # label positions
        width = 0.35  # width of the bars

        fig, ax = plt.subplots(figsize=(12,5))
        _ = ax.bar(x - width/2, df_train['total_count'], width, label='Train', color='skyblue')
        _ = ax.bar(x + width/2, df_val['total_count'], width, label='Validation', color='orange')

        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45)
        ax.set_ylabel("Total Count")
        ax.set_title("Total Count per Class: Train vs Validation")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)


        fig, ax = plt.subplots(figsize=(12,5))
        _ = ax.bar(x - width/2, df_train['mean_area'], width, label='Train', color='green')
        _ = ax.bar(x + width/2, df_val['mean_area'], width, label='Validation', color='red')

        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45)
        ax.set_ylabel("Mean Area")
        ax.set_title("Mean Area per Class: Train vs Validation")
        ax.legend()
        st.pyplot(fig)
        plt.close()
        st.write("The train class has a smaller average area in the validation set")
        st.write("But considering the small number of samples, this differnce is expected")
        st.write("Otherwise, all other classes except bus have more or less the same average size in both splits")

        num_train_images = 69863
        num_val_images = 10000

        # Compute per-image occurrence
        train_per_image = df_train['total_count'] / num_train_images
        val_per_image = df_val['total_count'] / num_val_images

        # Plot side-by-side bar chart
        x = np.arange(len(df_train.index))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12,5))
        _ = ax.bar(x - width/2, train_per_image, width, label='Train', color='skyblue')
        _ = ax.bar(x + width/2, val_per_image, width, label='Validation', color='orange')

        ax.set_xticks(x)
        ax.set_xticklabels(df_train.index, rotation=45)
        ax.set_ylabel("Average Occurrence per Image")
        ax.set_title("Per-Image Class Occurrence: Train vs Validation")
        ax.legend()
        st.pyplot(fig)
        plt.close()
        st.write("It can be seen that both splits have a similar average number of class occurences per image")

    def render_barplots_general(self):
        """
        Renders general statistics like category count and mean area
        for each of the dataset splits side by side for comparison
        """
        for split in self.splits:
            df = pd.read_csv(f"{self.csv_root}/{split}_stats.csv", index_col=0)
            df = df.T
            # --- Total Count per Class ---
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(df.index, df['total_count'], color='skyblue')
            ax.set_xlabel("Category")
            ax.set_ylabel("Total Count")
            ax.set_title(f"Total Count per Class - {split} set")
            ax.tick_params(axis="x", rotation=45)
            st.pyplot(fig)
            plt.close(fig)
            st.write("It looks like car is the category with the most number of occurences, followed by traffic sign, and then by traffic light.")
            st.write("person class also has a significant presence in the dataset.")
            st.write("train, motor, and rider are have the three lowest counts.") 
            st.write("I hypothesize from this information, that the model (if trained on this dataset) will perform well if tasked with detecting cars, but will not perform so well when tasked with detecting the three classes with the lowest counts")

    def render_barplots_useless(self):
        """
        Renders useless statistics like category count and mean area
        for each of the dataset splits side by side for comparison
        """
        for split in self.splits:            
            df = pd.read_csv(f"{self.csv_root}/{split}_stats.csv", index_col=0)
            df = df.T
            # --- Mean Area per Class ---
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(df.index, df['mean_area'], color='orange')
            ax.set_xlabel("Category")
            ax.set_ylabel("Mean Area")
            ax.set_title(f"Mean Area per Class - {split} set")
            ax.tick_params(axis="x", rotation=45)
            st.pyplot(fig)
            plt.close(fig)

            # --- Mean Width vs Mean Height ---
            x = np.arange(len(df.index))
            width = 0.35
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.bar(x - width/2, df['mean_width'], width, label='Mean Width')
            ax.bar(x + width/2, df['mean_height'], width, label='Mean Height')

            ax.set_xticks(x)
            ax.set_xticklabels(df.index, rotation=45)
            ax.set_xlabel("Category")
            ax.set_ylabel("Pixels")
            ax.set_title(f"Mean Width vs Mean Height - {split} set")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
        st.write("From these four plots on bbox size, I infer that both splits have a similar distriution of the average area")
        st.write("Also, the mean width and mean height do not vary that much. This means that the boxes have uniform dimensions on average")
        st.write("Having similar uniform dimensions is an important quality for training single-stage detectors like YOLO")

    def poulate_centers(self, split):
        """
        Compute the bbox centers and save them
        """
        labels_path = f"{self.labels_root}/bdd100k_labels_images_{split}.json"
        with open(labels_path, "r") as f:
            for obj in ijson.items(f, "item"):
                for label in obj["labels"]:
                    if label["category"] in self.categories:
                        box = label["box2d"]
                        x_center = (box["x1"] + box["x2"]) / 2
                        y_center = (box["y1"] + box["y2"]) / 2
                        self.box_centers[label["category"]].append((x_center, y_center))

    def poulate_areas(self, split):
        """
        Compute the bounding box areas for all instances,
        normalize them, and save them
        """
        labels_path = f"{self.labels_root}/bdd100k_labels_images_{split}.json"
        with open(labels_path, "r") as f:
            for obj in ijson.items(f, "item"):
                for label in obj["labels"]:
                    if label["category"] in self.categories:
                        box = label["box2d"]
                        box_area = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
                        normalized_box_area = box_area / self.image_height / self.image_width
                        self.box_sizes[label["category"]].append(normalized_box_area)

    def generate_heatmaps(self, bins=50):
        """
        Generates heatmaps for each class to visualize the
        distribution of bounding boxes based on their centers
        """
        for split in self.splits:
            self.reset_box_centers()
            self.poulate_centers(split)

            st.write(f"Here is what the class-wise heatmaps look like for the {split} set")
            fig, axes = plt.subplots(5, 2, figsize=(14, 25))  # grid of 10 plots
            axes = axes.ravel()  # flatten for easy indexing

            for idx, category in enumerate(self.categories):
                centers = np.array(self.box_centers[category], dtype=float)
                x_coords, y_coords = centers[:, 0], centers[:, 1]
                heatmap, _, _ = np.histogram2d(x_coords, y_coords, bins=bins)

                im = axes[idx].imshow(
                    heatmap.T,
                    origin='upper',  # BDD100K coordinate system
                    cmap='hot',
                    interpolation='nearest',
                    extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]
                )
                fig.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
                axes[idx].set_title(f"BBox Centers for {category} in {split} set")
                axes[idx].set_xlabel("X")
                axes[idx].set_ylabel("Y")

            # Hide unused subplots if fewer than 10 categories
            for j in range(len(self.categories), len(axes)):
                axes[j].axis("off")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    def generate_boxplot(self):
        """
        Computes bbox areas for each split
        and renders a boxplot on streamlit
        """
        for split in self.splits:
            self.reset_box_sizes()
            self.poulate_areas(split)
            df = pd.DataFrame([
                {"class": cls, "value": v}
                for cls, values in self.box_sizes.items()
                for v in values
            ])
            fig, ax = plt.subplots(figsize=(6,4))
            sns.boxplot(x="class", y="value", data=df, ax=ax)
            ax.set_title(f"Per-class Boxplot of BBox Area - {split}")
            ax.set_xlabel("Class")
            ax.set_ylabel("Area")
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
            plt.close(fig)