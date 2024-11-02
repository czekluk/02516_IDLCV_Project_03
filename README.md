# 02516_IDLCV_Project_03

## Project Description

This project aims at implementing an object detection network based on the Potholes dataset. The project has multiple parts which are:

- Generating feature proposals
- Classifying proposed regions using classification network
- possibly even more :)

## Data

The pothole dataset consists of 665 images with pothole annotations in the form of bounding boxes. The annotations are stored in an XML PascalVOC-style format.

![pothole](docs/img-1.jpg)

## Run training on the HPC

To start training using batch jobs first modify the jobscript.sh file you want to use. Please note that every contributor is recommended to create his own jobscript_[NAME].sh file according to his preferences.

Then execute:

```bash
bsub -app c02516_1g.10gb < jobscript_[NAME].sh
```

To monitor the progress execute:
```bash
bstat
```

To abort the run:

```bash
bkill <job-id>
```

## ToDo

- Visualisation of data (bounding boxes + images) - Nandor
- Dataloaders + preprocessing (class imbalance!) - Nandor
- Object proposal extractors - Alex
- Object proposal preparation - Alex
- Metrics (for models & proposals) - Alex
- Training loop - Zeljko
- Training visualisations - Zeljko
- CNN classifier (N+1 classes) - Filip
- training, training, training - Filip
- evaluation of classification accuracy, object detection accuracy, etc. - Lukas

Next meeting on Saturday (02.11.2024) at 18:00.

## ToDo V2

Next meeting on Wednesday (06.11.2024) during lecture.

## Submission Deadline

Submission on Tuesday 19.11.2024 at 22:00.

## Poster 

The poster can be found [here](https://dtudk.sharepoint.com/:p:/r/sites/IntroDLCV2024/Delte%20dokumenter/General/Poster_Project_03.pptx?d=wc5ff158f0d86492ca802f3dc20e749ef&csf=1&web=1&e=V9CkQC).
