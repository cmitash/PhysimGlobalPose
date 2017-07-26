#!/bin/bash
roscore&
source $PHYSIM_GLOBAL_POSE/devel/setup.sh
$PHYSIM_GLOBAL_POSE/src/detection_package/bin/detect_bbox&