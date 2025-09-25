## Task 2: Model Selection

I have picked Faster R-CNN as my model of choice. I have also included a notebook where I trained a pre-trained Faster R-CNN with a ResNet50 backbone and Feature Pyramid Network for 5 epochs on the entire dataset.  

Please refer to the Readme in /task2_model_selection_and_training folder for details.

Additionally, I have included a notebook in /task2_model_selection_and_training/YOLO, where I trained YOLO11 for 5 epochs on Google Colab. The notebook can be run in Colab as-is, provided that you have the assignment_data_bdd.zip file on your Google drive storage.  

While I did not pick YOLO11 as my model of choice, I left the notebook in to showcase how I converted the labels by hand to Ultralytics format. Since the Ultralytics API abstracts away the process of writing a training loop and dataloader, I decided to go with Faster R-CNN.

# 2.1 Why Faster R-CNN with ResNet50 and FPN

- Faster R-CNN is a two-stage detector where one part of the model architecture, the Region Proposal Network, focuses on where to look, whereas the classification head decides what the class is. This two stage detection process helps it detect object classes and their bounding boxes with high accuracy.
- The Feature Pyramid Network (FPN) helps in accurately identifying objects at multiple scales. This is where a single stage detector like YOLO might struggle.
- The ResNet50 backbone provides a rich set of features which are easily trainable thanks to the skip connections in resnet modules.
- The network provides an end-to-end multitask loss which can be used to optimize the weights in a simple training loop.
- Faster R-CNN is a mature model that has been around for a while, and the available pretrained networks provided by torchvision are good enough and trained on COCO, so that means that the evaluation task won't be hard because the model will hopefully not output garbage.

# 2.2 Training Faster R-CNN

Please make sure to check the notebook in this directory called 2.Model_Selection_and_Training.ipynb

Thank you very much!