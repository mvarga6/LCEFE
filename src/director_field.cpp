#include "director_field.h"

UniformField::UniformField(const float theta, const float phi)
{
	this->const_director.theta = theta;
	this->const_director.phi = phi;
}


DirectorOrientation UniformField::GetDirectorAt(
	const float x, 
	const float y, 
	const float z)
{
	// we always return the same thing no matter
	// what x, y, z values are passed
	return this->const_director;
}


CartesianDirectorField::CartesianDirectorField(
	ScalerField3D *thetaField, 
	ScalerField3D *phiField)
{
	this->theta_field = thetaField;
	this->phi_field = phiField;
}



DirectorOrientation CartesianDirectorField::GetDirectorAt(
	const float x,
	const float y,
	const float z)
{
	DirectorOrientation result;
	result.theta = (*theta_field)(x, y, z);
	result.phi = (*phi_field)(x, y, z);
	return result;
}
