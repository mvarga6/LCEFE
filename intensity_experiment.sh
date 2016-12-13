#!/bin/bash
DATADIR="/nfs/02/ksu0236/mvarga/LCEOUT"
[ -d $DATADIR ] || mkdir $DATADIR
cd $DATADIR

for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
	qsub -v datadir=$DATADIR,son=-0.1,soff=0.1,smin=-${i},phi=80,out=planartop_I${i}_phi80 ~/mvarga/LCEFE/myscript.job
	#qsub -v datadir=$DATADIR,flags=--homeotop,phi=${p},smax=0,smin=-1.0,sqh=0.06,sql=0.985,out=homeotop_phi${p} ~/mvarga/LCEFE/myscript.job
done
