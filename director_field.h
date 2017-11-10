#ifndef __DIRECTOR_FIELD_H__
#define __DIRECTOR_FIELD_H__

#include "functions.hpp"

struct DirectorOrientation
{
	real theta, phi;
};

/*
  A based director field
*/
class DirectorField
{
public:
	virtual DirectorOrientation GetDirectorAt(const real, const real, const real) = 0;	
};

/*
  A director field that is uniform everywhere
*/
class UniformField : public DirectorField
{
	DirectorOrientation const_director;
public:
	UniformField(const real theta, const real phi);
	DirectorOrientation GetDirectorAt(const real x, const real y, const real z);
};



/*
  A director field that has a gradient
  in theta and/or phi
*/
class CartesianDirectorField : public DirectorField
{
	ScalerField3D *theta_field, *phi_field;
		
public: 
	CartesianDirectorField(ScalerField3D *thetaField, ScalerField3D *phiField);
	DirectorOrientation GetDirectorAt(const real x, const real y, const real z);
};

#endif
