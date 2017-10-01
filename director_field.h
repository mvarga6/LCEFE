#ifndef __DIRECTOR_FIELD_H__
#define __DIRECTOR_FIELD_H__

#include "functions.hpp"

struct DirectorOrientation
{
	float theta, phi;
};

/*
  A based director field
*/
class DirectorField
{
public:
	virtual DirectorOrientation GetDirectorAt(const float, const float, const float) = 0;	
};

/*
  A director field that is uniform everywhere
*/
class UniformField : public DirectorField
{
	DirectorOrientation const_director;
public:
	UniformField(const float theta, const float phi);
	DirectorOrientation GetDirectorAt(const float x, const float y, const float z);
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
	DirectorOrientation GetDirectorAt(const float x, const float y, const float z);
};

#endif
