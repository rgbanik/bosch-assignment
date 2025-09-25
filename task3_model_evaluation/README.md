## Task 3

In this task, I have taken the model which I had trained for 5 epochs in task 2, and evaluated it's performance on the provided validation set.

Please refer to the notebook named 3.Model_Evaluation.ipynb in this directory.

However, it was only while attempting this task did I realize that I had trained with incorrect class ids.
Faster R-CNN requires 1-indexed labels because 0 is for background. However, I had used 0-indexed labels, and my COCO evaluation resutls are a bit off.

Nevertheless, the qualitative results show that the model is rather good at detecting cars, traffic lights, trucks, and even trains.
It is surprising that it is so good at identifying trains, because there were not many train images in the dataset.

The quantitative results show an improvement over the past epochs, but the results for some classes really can't be trusted due to the index shift. I wonder why COCO has to be so strict about it, and why the mapping can't simply be adjusted after training.

Thank you for checking my submission, I really appreciate it!

Looking forward to hearing from you soon! Best Regards!
