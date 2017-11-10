#ifndef __FUNCTIONS_HPP__
#define __FUNCTIONS_HPP__

#include "defines.h"
#include <math.h>
#include <vector>
#include "parameters.h"

using namespace std;

class UnivariableFunction { public:	virtual real operator()(real) = 0; };
class ScalerField3D{ public: virtual real operator()(real, real, real) = 0; };


class ConstantField3D : public ScalerField3D
{
	real a;
public:
	inline ConstantField3D(real A)
	{
		a = A;
	}
	
	inline real operator()(real x, real y, real z)
	{
		return a;
	}
};

template <int D = 1>
class Polynomial : public UnivariableFunction
{
	real c[D];
	real c0;
public:
	
	inline Polynomial(vector<real> coef, real C0 = 0)
	{
		// make coef vector of correct length
		if (coef.size() < D)
		{
			for(int i = 0; i < D - coef.size(); i++)
			{
				coef.push_back(0.0f);
			}
		}
		
		// assign coefficents
		for(int d = 0; d < D; d++)
		{
			c[d] = coef[d];
		}		
		c0 = C0;
	}
	
	inline real operator()(real u)
	{
		real result = c0;
		for(int d = 0; d < D; d++)
		{
			result += c[d] * powf(u, d+1);
		}
		return result;
	}
};


class UnityFunction : public UnivariableFunction
{
public:
	inline real operator()(real u)
	{
		return 1.0f;
	}
};

class ZeroFunction : public UnivariableFunction
{
public:
	inline real operator()(real u)
	{
		return 0.0f;
	}
};

class IdenityFunction : public UnivariableFunction
{
public:
	inline real operator()(real u)
	{
		return u;
	}
};

class Sinusoinal : public UnivariableFunction
{
	real factor, phase;
public:
	inline Sinusoinal(real wavelength = _2PI, real phase = 0)
	{
		this->factor = (2.0f * PI) / wavelength;
		this->phase = phase;
	}
	
	inline real operator()(real u)
	{
		return sinf(u * factor + phase);
	}
};

typedef Polynomial<1> Linear;
typedef Polynomial<2> Quadradic;
typedef Polynomial<3> Cubic;
typedef Polynomial<4> Quartic;
typedef Polynomial<5> Quintic;

class MultiplicativeField3D : public ScalerField3D
{
	UnivariableFunction *fofx, *fofy, *fofz;
public:
	inline MultiplicativeField3D(UnivariableFunction *FofX, UnivariableFunction *FofY = NULL, UnivariableFunction *FofZ = NULL)
	{
		fofx = (FofX != NULL ? FofX : new UnityFunction);
		fofy = (FofY != NULL ? FofY : new UnityFunction);
		fofz = (FofZ != NULL ? FofZ : new UnityFunction);
	}
	
	inline real operator()(real x, real y, real z)
	{
		return (*fofx)(x) * (*fofy)(y) * (*fofz)(z);
	}
};

class AdditiveField3D : public ScalerField3D
{
	UnivariableFunction *fofx, *fofy, *fofz;
public:
	inline AdditiveField3D(UnivariableFunction *FofX, UnivariableFunction *FofY = NULL, UnivariableFunction *FofZ = NULL)
	{
		fofx = (FofX != NULL ? FofX : new ZeroFunction);
		fofy = (FofY != NULL ? FofY : new ZeroFunction);
		fofz = (FofZ != NULL ? FofZ : new ZeroFunction);
	}
	
	inline real operator()(real x, real y, real z)
	{
		return (*fofx)(x) + (*fofy)(y) + (*fofz)(z);
	}
};

#endif
