#!/bin/bash
DATADIR="/nfs/02/ksu0236/mvarga/LCEOUT"
[ -d $DATADIR ] || mkdir $DATADIR
cd $DATADIR

# does nothing right now

for rate in 0.02 0.04 0.06 0.08 0.1 0.12 0.14 0.16 0.18 0.20 0.22 0.24 0.26 0.28 0.3 0.32 0.34 0.36 0.38 0.4 0.45 0.5
do
	#qsub -v datadir=$DATADIR,son=-$rate,soff=$rate,phi=80,out=planartop_phi80_R${rate} ~/mvarga/LCEFE/myscript.job
	#qsub -v datadir=$DATADIR,flags=--homeotop,phi=${p},smax=0,smin=-1.0,sqh=0.06,sql=0.985,out=homeotop_phi${p} ~/mvarga/LCEFE/myscript.job
done
