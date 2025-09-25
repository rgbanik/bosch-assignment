# bosch-assignment

Hope you are doing well, and thank you for evaluating my submission!


## Task 1: Dataset Analysis
Please clone this repository and cd into task1_data_analysis by executing the following commands:

```
git clone git@github.com:rgbanik/bosch-assignment.git
cd bosch-assignment/task1_data_analysis
```

### Known issue: as I have edited the files on Windows with VSCode, you will probably need to run the following commands to prevent the error: /bin/bash: line 1: /app/restructure_dataset.sh: cannot execute: required file not found
Please enter wsl if on Windows, as these are Linux commands
```
sudo apt install dos2unix
```
```
dos2unix restructure_dataset.sh
```
This will fix the line formatting in case it is still CRLF for some reason

### Building and running the container
You will need to build the docker image by running:

```
docker build -t assignment .
```

Once the image has been built, please run the container by replacing the text in <> with the path to the EXTRACTED dataset on your local system:

```
# On Linux
docker run -it --rm -p 8501:8501 -v </path/to/assignment_data_bdd>:/data assignment
```

```
# On Windows
docker run -it --rm -p 8501:8501 --mount type=bind,source="<C:\path\to\assignment_data_bdd>",target=/data assignment
```

On Linux, please then navigate to http://0.0.0.0:8501/  
On Windows, please navigate to http://localhost:8501/

You will be greeted by some text in the webpage, until the analysis and visualization modules start rendering to the webpage directly. This may take a couple of minutes.

### Please note: It is important that the dataset is EXTRACTED and port 8501 is free. For example, if you extract assignment_data_bdd.zip to /home/username/content/assignment_data/, please execute: docker run -it --rm -p 8501:8501 -v /home/username/content/assignment_data:/data assignment

If the incorrect drive is mounted, the app will not start. As a just-in-case, I have uploaded the printed analysis off of streamlit to the repo as /task1_data_analysis/task1_data_analysis.pdf

Insights gained from analysis: 
- There are very few instances of train, rider, and motorbike
- There are several instances of cars: visible, truncated, and occluded
- There average bounding box area and class counts are similar in both splits for all classes
- The train class has several outliers that cover over 60 percent of the image, with some covering almost the entire image
- The most frequent classes are also more uniformly distributed accross the dataset, whereas the infrequent classes are haphazardly scattered across the dataset.

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

## Task 3

In this task, I have taken the model which I had trained for 5 epochs in task 2, and evaluated it's performance on the provided validation set.

Please refer to the notebook named 3.Model_Evaluation.ipynb in the /task3_model_evaluation directory.

However, it was only while attempting this task did I realize that I had trained with incorrect class ids.
Faster R-CNN requires 1-indexed labels because 0 is for background. However, I had used 0-indexed labels, and my COCO evaluation resutls are a bit off.

Nevertheless, the qualitative results show that the model is rather good at detecting cars, traffic lights, trucks, and even trains.
It is surprising that it is so good at identifying trains, because there were not many train images in the dataset.

The quantitative results show an improvement over the past epochs, but the results for some classes really can't be trusted due to the index shift. I wonder why COCO has to be so strict about it, and why the mapping can't simply be adjusted after training.

Thank you for checking my submission, I really appreciate it!

Looking forward to hearing from you soon! Best Regards!
