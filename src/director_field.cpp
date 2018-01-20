#include "director_field.h"
#include "genrand.h"
#include <stdio.h>
#include <math.h>
#include <time.h>

UniformField::UniformField(const real theta, const real phi)
{
	this->const_director.theta = theta;
	this->const_director.phi = phi;
}


DirectorOrientation UniformField::GetDirectorAt(
	const real x, 
	const real y, 
	const real z)
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
	const real x,
	const real y,
	const real z)
{
	DirectorOrientation result;
	result.theta = (*theta_field)(x, y, z);
	result.phi = (*phi_field)(x, y, z);
	return result;
}

RadialDirectorField::RadialDirectorField(const float3 origin)
 : origin(origin)
{
}

DirectorOrientation RadialDirectorField::GetDirectorAt(
	const real x,
	const real y, 
	const real z)
{
	DirectorOrientation result;
	const real dx = x - origin.x;
	const real dy = y - origin.y;
	const real dz = z - origin.z;
	result.theta = atan2(sqrt(dx*dx + dy*dy), dz);
	result.phi = atan2(dy, dx);
	return result;
}

RandomDirectorField::RandomDirectorField()
{ 
	srand(time(NULL));    //seed random number generator
	mt_init();       //initialize random number generator
	purge();         //free up memory in random number generator
}

DirectorOrientation RandomDirectorField::GetDirectorAt(
	const real x,
	const real y, 
	const real z)
{
	DirectorOrientation result;
	result.theta = genrand() * M_PI;
	result.phi = genrand() * 2.0 * M_PI;
	return result;
}