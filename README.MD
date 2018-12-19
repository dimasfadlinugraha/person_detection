# Person Detection

## Description
A Simple Person Detection Program  

This program uses two algorithm for person detection and you can choose between this two :
* Haarcascade Upper Body
* HOG Descriptor (People Detector)  

Also, this program uses Kalman Filter and Custom Tracker Class to predict and track person

## How To Run
This program contain two mode :
* `HOG`
* `HAARCASCADE` -> **Default**

To choose between mode you need to provide mode name to **os environment** before running the program and you also need to provide video input name. Here is an example to run the program : 

```MODE=HAARCASCADE python person_detection.py ```

## Prerequisites
* Python 2.7
* Opencv 3.4

## Credits
* [Srini Ananthakrishnan](https://github.com/srianant/kalman_filter_multi_object_tracking) for Kalman Filter and Tracker. I use a modified version of Kalman Filter and Tracker from that project to suite my need.

## Side Notes
* Change the input video