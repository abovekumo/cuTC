#!/bin/bash

testdir="/home/dm/database/test/"
traindir="/home/dm/database/train/"
validatedir="/home/dm/database/validate/"

algtype=sgd2
tensorname="nell-2"
trainname=_tr
trainnamen=_tr2
trainnamenn=_tr3
testname=_te
validatename=_v
echo $traindir$tensorname$trainname.tns $traindir$tensorname$trainnamen.tns $traindir$tensorname$trainnamenn.tns $validatedir$tensorname$validatename.tns $testdir$tensorname$testname.tns 

./tc $traindir$tensorname$trainname.tns $traindir$tensorname$trainnamen.tns $traindir$tensorname$trainnamenn.tns $validatedir$tensorname$validatename.tns $testdir$tensorname$testname.tns > ./log/$algtype$tensorname.log 2>&1
