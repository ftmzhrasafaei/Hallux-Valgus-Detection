# Hallux Valgus Detection
Hallux Valgus Detection with CNN

This program uses a dataset (collected by my colleague and me, labeld by kind medical professors of Shahid Beheshti University) to predict the Hallux Valgus, a common foot deformity. 

You can access to the dataset, using this link to my GoogleDrive: https://drive.google.com/drive/folders/1ROlOI9OjOuUoJ-Voru3-nbB7vjsFZtXb?usp=sharing

I used Keras models to create the CNN for this matter. My model got an accuracy between 70 to 80 percent and f-score between 60 to 70 on validation set.

The dataset was unbalanced so I also applied some augmentation on the smaller class. Thas is, I added a rotated version of each images. 

All the plots, loss, accuracy, fscore, will be shown once you run the program. The program also shows 5 predictions as examples.

