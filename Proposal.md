# PASS THE SALT

## The Problem

Subsurface salt is a blessing and a curse for Oil and Gas exploration and production. In one hand salt intrusions help creating and sealing oil and gas traps that can be very prolific. In the other hand there are many hazards related to drilling through or near salt domes.

Reflection Seismic Imaging is used to determine structural and stratigraphic characteristics of the subsurface, it works by recording sound waves as they reflect from the different geological interfaces they found along their path. This technique relies heavily on some properties of the velocities at which the sound travels through the rocks. Salt however, stretches those assumptions reducing the ability for geophysicists to properly image them precisely.

The goal of this project is to precisely determine what is Salt and what isn't in seismic data.

## The data

The data-set consists of 4000 images which have been already interpreted and have 4000 corresponding masks identifying salt and non-salt, and 18000 non classified images as test set. The origin of the data set is a [Kaggle competition](https://www.kaggle.com/c/tgs-salt-identification-challenge) sponsored by [TGS](http://www.tgs.com)


## Deliverable

A functional classificator for Salt or Not-Salt categories. Ideally with a CNN but also explore simpler algorithms.
