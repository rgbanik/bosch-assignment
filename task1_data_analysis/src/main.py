import os
import streamlit as st
from pathlib import Path
import ijson

def run_analysis():
    dataset_root = Path(os.environ.get("DATASET_PATH", "/data"))
    labels_path = dataset_root / "bdd100k_labels_release" / "bdd100k" / "labels"
    val_labels = labels_path / "bdd100k_labels_images_val.json"
    train_labels = labels_path / "bdd100k_labels_images_train.json"

    
    val_labels_dict = {}
    train_labels_dict = {}

    with val_labels.open("r", encoding="utf-8") as f:
        for obj in ijson.items(f, "item"):
            for label in obj["labels"]:
                name = label["category"]
                if name not in ["lane", "drivable area"]:
                    if val_labels_dict.get(name) is None:
                        st.write(name)
                        val_labels_dict[name] = 1
                    else:
                        val_labels_dict[name] += 1

        st.write("FINAL LABELS VAL:")
        st.write(val_labels_dict)

    with train_labels.open("r", encoding="utf-8") as f:
        for obj in ijson.items(f, "item"):
            for label in obj["labels"]:
                name = label["category"]
                if name not in ["lane", "drivable area"]:
                    if train_labels_dict.get(name) is None:
                        st.write(name)
                        train_labels_dict[name] = 1
                    else:
                        train_labels_dict[name] += 1

        st.write("FINAL LABELS TRAIN:")
        st.write(train_labels_dict)

if __name__ == "__main__":
    run_analysis()