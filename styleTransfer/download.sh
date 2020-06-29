#!/bin/bash

BASE_URL="http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/"

cd models
wget -N --no-check-certificate "$BASE_URL/instance_norm/candy.t7"
wget -N --no-check-certificate "$BASE_URL/instance_norm/la_muse.t7"
wget -N --no-check-certificate "$BASE_URL/instance_norm/mosaic.t7"
wget -N --no-check-certificate "$BASE_URL/instance_norm/feathers.t7"
wget -N --no-check-certificate "$BASE_URL/instance_norm/the_scream.t7"
wget -N --no-check-certificate "$BASE_URL/instance_norm/udnie.t7"