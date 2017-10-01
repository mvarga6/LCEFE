#include "director_field.h"

UniformField::UniformField(const float theta, const float phi)
{
	this->const_director.theta = theta;
	this->const_director.phi = phi;
}


DirectorOrientation UniformField::GetDirectorAt(const float x, const float y, const float z)
{
	// we always return the same thing no matter
	// what x, y, z values are passed
	return this->const_director;
}
