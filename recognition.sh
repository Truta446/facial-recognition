#!/bin/bash

# Export url to use on capture
export URL=http://192.168.0.101:9443/videofeed

# Start application
# python recognition_eigenfaces.py
# python recognition_fisherfaces.py
python recognition_lbph.py