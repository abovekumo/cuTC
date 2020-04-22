#!/bin/python3

#to generate the training and validation dataset
import random
import sys
from operator import *

tensordir = "../database/"

tensorname = "matmul_403"

otensor = open(tensordir+tensorname+".tns","r")
ttensor = open(tensordir+tensorname+"_tr.tns","w")
vtensor = open(tensordir+tensorname+"_v.tns","w")
test = open(tensordir+tensorname+"_te.tns","w")

train = []
testlist = []
validate = []
original = []

percentage = 0.8

mini = sys.maxsize
minj = sys.maxsize
mink = sys.maxsize
maxi = 0
maxj = 0
maxk = 0

print("begin original\n")

for each in otensor:
    otensorb = []
    anewline = each.split(' ')
    otensorb.append(int(anewline[0]))
    otensorb.append(int(anewline[1]))
    otensorb.append(int(anewline[2]))
    otensorb.append(float(anewline[3]))
    if(otensorb[0]>maxi):
        maxi = otensorb[0]
    if(otensorb[1]>maxj):
        maxj = otensorb[1]
    if(otensorb[2]>maxk):
        maxk = otensorb[2]
    if(otensorb[0]<mini):
        mini = otensorb[0]
    if(otensorb[1]<minj):
        minj = otensorb[1]
    if(otensorb[2]<mink):
        mink = otensorb[2]
    original.append(otensorb)

print("finish original\n")

original.sort(key=itemgetter(0,1))

print(maxi,mini)
print('\n')
print(maxj,minj)
print('\n')
print(maxk,mink)
print('\n')

for each in original:
    flag = random.randint(0,9)
    if(flag<8):
      train.append(each)
    else:
      if(flag == 8):
          testlist.append(each)
      else:
          validate.append(each)

for eachline in train:
    ttensor.write("%d %d %d %.1f\n"%(eachline[0],eachline[1],eachline[2],eachline[3]))

for eachline in validate:
    vtensor.write("%d %d %d %.1f\n"%(eachline[0],eachline[1],eachline[2],eachline[3]))

for eachline in testlist:
    test.write("%d %d %d %.1f\n"%(eachline[0],eachline[1],eachline[2],eachline[3]))

otensor.close()
ttensor.close()
vtensor.close()
test.close()
