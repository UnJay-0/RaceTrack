#!/bin/bash
track_number=$1
python -m main "tracks/track_0${track_number}.t"
perl visualise.pl tracks/"track_0${track_number}.t" output/"track_0${track_number}"/"track_0${track_number}_trip.csv" output/"track_0${track_number}"/"track_0${track_number}"
pdflatex -output-directory output/"track_0${track_number}" output/"track_0${track_number}"/"track_0${track_number}"
