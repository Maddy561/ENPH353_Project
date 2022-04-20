# ENPH353_Project
Final project for ENPH353. This repo contains the code to detect and return license plates (in my_controller) as well as the notebooks used for training the neural network. 
![Software Architecture](https://user-images.githubusercontent.com/68255880/164309155-09600464-33eb-4ef2-87a6-e4d5b3066329.jpg)

The node used to detect license plates subscribes to the in-simulation camera and publishes to the license plate topic. 
![ROS Nodes and Topics](https://user-images.githubusercontent.com/68255880/164309221-b92cab61-83d8-423e-b14b-24c7d0743261.jpg)

The data for the neural network was created and augmented in google colab. The network itself was created in colab and then saved and downloaded locally.
