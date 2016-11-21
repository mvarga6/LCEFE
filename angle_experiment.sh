#!/bin/bash
DATADIR="/nfs/02/ksu0236/mvarga/LCEOUT"
[ -d $DATADIR ] || mkdir $DATADIR
cd $DATADIR

for p in 90 85 80 75 70 65 60 55 50
do
	qsub -v phi=${p},smax=0,smin=-1.0,sqh=0.18,sql=0.925,out=A_phi${p} ~/mvarga/LCEFE/myscript.job
done
