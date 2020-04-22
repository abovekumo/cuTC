#!/bin/python3

#to generate the training and validation dataset
import random
import sys
from operator import *

tensordir = "../database/train/"

tensorname = "matmul_403"

ttensor = open(tensordir+tensorname+"_tr.tns","r")
ttensor2 = open(tensordir+tensorname+"_tr2.tns", "w")
ttensor3 = open(tensordir+tensorname+"_tr3.tns", "w")


train1 = []
train2 = []

maxi = 0
maxj = 0
maxk = 0

for each in ttensor:
    ttensorb = []
    anewline = each.split(' ')
    ttensorb.append(int(anewline[0]))
    ttensorb.append(int(anewline[1]))
    ttensorb.append(int(anewline[2]))
    ttensorb.append(float(anewline[3]))
    train1.append(ttensorb)
    train2.append(ttensorb)

print("finish reading data\n")

train1.sort(key=itemgetter(1,2))
train2.sort(key=itemgetter(2,0))

for eachline in train1:
    ttensor2.write("%d %d %d %.1f\n"%(eachline[0],eachline[1],eachline[2],eachline[3]))

for eachline in train2:
    ttensor3.write("%d %d %d %.1f\n"%(eachline[0],eachline[1],eachline[2],eachline[3]))

ttensor.close()
ttensor2.close()
ttensor3.close()
