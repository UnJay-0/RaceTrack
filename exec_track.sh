#!/bin/bash
track_number=$1
construction_type=$2
python -m main "tracks/track_0${track_number}.t" "${construction_type}"
# step
perl visualise.pl tracks/"track_0${track_number}.t" output/"track_0${track_number}"/"${construction_type}"/"track_0${track_number}_trip.csv" output/"track_0${track_number}"/"${construction_type}"/"track_0${track_number}"
pdflatex -output-directory output/"track_0${track_number}"/"${construction_type}" output/"track_0${track_number}"/"${construction_type}"/"track_0${track_number}"

#improved
perl visualise.pl tracks/"track_0${track_number}.t" output/"track_0${track_number}"/"improved"/"track_0${track_number}_trip.csv" output/"track_0${track_number}"/"improved"/"track_0${track_number}"
pdflatex -output-directory output/"track_0${track_number}"/"improved" output/"track_0${track_number}"/"improved"/"track_0${track_number}"
