#ifndef __DIRECTOR_FIELD_H__
#define __DIRECTOR_FIELD_H__

#include "functions.hpp"

struct DirectorOrientation
{
	real theta, phi;
};

///
/// A based director field
class DirectorField
{
public:

	///
	/// Returns the director orientations and a given
	/// position in Euclidean space
	virtual DirectorOrientation GetDirectorAt(const real, const real, const real) = 0;	
};

///
/// A director field that is uniform everywhere
class UniformField : public DirectorField
{
	DirectorOrientation const_director;
public:

	///
	/// Cosntruct with theta and phi value of
	/// the constant directory field
	UniformField(const real theta, const real phi);

	///
	/// Returns the director orientations and a given
	/// position in Euclidean space
	DirectorOrientation GetDirectorAt(const real x, const real y, const real z);
};



///
/// A director field that has a gradient
/// in theta and/or phi
class CartesianDirectorField : public DirectorField
{
	ScalerField3D *theta_field, *phi_field;
		
public:

	///
	/// Create a director field with theta(x,y,z) and phi(x,y,z)
	CartesianDirectorField(ScalerField3D *thetaField, ScalerField3D *phiField);

	///
	/// Returns the director orientations and a given
	/// position in Euclidean space	
	DirectorOrientation GetDirectorAt(const real x, const real y, const real z);
};

///
/// Given an origin, create a director field
/// radial with reference to the origin.
class RadialDirectorField : public DirectorField
{
	const float3 origin;
public:

	///
	/// Construct a director field that points
	/// radially out from the given origin
	RadialDirectorField(const float3 origin);

	///
	/// Returns the director orientations and a given
	/// position in Euclidean space	
	DirectorOrientation GetDirectorAt(const real x, const real y, const real z);
};

#endif
