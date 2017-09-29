#!/bin/bash
DATADIR="/nfs/02/ksu0236/mvarga/LCEOUT"
[ -d $DATADIR ] || mkdir $DATADIR
cd $DATADIR

for p in 90 87.5 85 82.5 80 77.5 75 72.5 70 67.5 65 60 55 50
do
	#qsub -v datadir=$DATADIR,phi=${p},smax=0,smin=-1.0,sqh=0.1,sql=0.925,out=planartop_phi${p} ~/mvarga/LCEFE/myscript.job
	qsub -v datadir=$DATADIR,flags=--homeotop,phi=${p},smax=0,smin=-1.0,sqh=0.06,sql=0.985,out=homeotop_phi${p} ~/mvarga/LCEFE/myscript.job
done
