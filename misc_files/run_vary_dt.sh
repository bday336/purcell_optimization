#!/bin/bash

screen -S l1l1l1a0 -dm python vary_dt_sims.py 1 1 1 0 .001 i
screen -S l1l1l1a0 -dm python vary_dt_sims.py 1 1 1 0 .0001 i
screen -S l1l1l1a0 -dm python vary_dt_sims.py 1 1 1 0 .00001 i

screen -S lp1lp1lp1a0 -dm python vary_dt_sims.py .1 .1 .1 0 .001 p
screen -S lp1lp1lp1a0 -dm python vary_dt_sims.py .1 .1 .1 0 .0001 p
screen -S lp1lp1lp1a0 -dm python vary_dt_sims.py .1 .1 .1 0 .00001 p

screen -S lp01lp01lp01a0 -dm python vary_dt_sims.py .01 .01 .01 0 .001 p
screen -S lp01lp01lp01a0 -dm python vary_dt_sims.py .01 .01 .01 0 .0001 p
screen -S lp01lp01lp01a0 -dm python vary_dt_sims.py .01 .01 .01 0 .00001 p

screen -S lp001lp001lp001a0 -dm python vary_dt_sims.py .001 .001 .001 0 .001 p
screen -S lp001lp001lp001a0 -dm python vary_dt_sims.py .001 .001 .001 0 .0001 p
screen -S lp001lp001lp001a0 -dm python vary_dt_sims.py .001 .001 .001 0 .00001 p

screen -S lp0001lp0001lp0001a0 -dm python vary_dt_sims.py .0001 .0001 .0001 0 .001 p
screen -S lp0001lp0001lp0001a0 -dm python vary_dt_sims.py .0001 .0001 .0001 0 .0001 p
screen -S lp0001lp0001lp0001a0 -dm python vary_dt_sims.py .0001 .0001 .0001 0 .00001 p

