#!/bin/bash
DATADIR="/nfs/02/ksu0236/mvarga/VTK"
[ -d $DATADIR ] || mkdir $DATADIR
cd $DATADIR

for p in 90 85 80 75 70 65 60 55 50
do
	qsub -v phi=$p,smax=0,smin=-0.5,sqh=0.15,sql=0.95,out=phi${p} ~/mvarga/LCEFE/myscript.job
done
