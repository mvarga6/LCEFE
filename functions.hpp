#ifndef __FUNCTIONS_HPP__
#define __FUNCTIONS_HPP__

#include <math.h>
#include <vector>
#include "parameters.h"

using namespace std;

class UnivariableFunction { public:	virtual float operator()(float) = 0; };
class ScalerField3D{ public: virtual float operator()(float, float, float) = 0; };


class ConstantField3D : public ScalerField3D
{
	float a;
public:
	inline ConstantField3D(float A)
	{
		a = A;
	}
	
	inline float operator()(float x, float y, float z)
	{
		return a;
	}
};

template <int D = 1>
class Polynomial : public UnivariableFunction
{
	float c[D];
	float c0;
public:
	
	inline Polynomial(vector<float> coef, float C0 = 0)
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
	
	inline float operator()(float u)
	{
		float result = c0;
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
	inline float operator()(float u)
	{
		return 1.0f;
	}
};

class ZeroFunction : public UnivariableFunction
{
public:
	inline float operator()(float u)
	{
		return 0.0f;
	}
};

class IdenityFunction : public UnivariableFunction
{
public:
	inline float operator()(float u)
	{
		return u;
	}
};

class Sinusoinal : public UnivariableFunction
{
	float factor, phase;
public:
	inline Sinusoinal(float wavelength = _2PI, float phase = 0)
	{
		this->factor = (2.0f * PI) / wavelength;
		this->phase = phase;
	}
	
	inline float operator()(float u)
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
	
	inline float operator()(float x, float y, float z)
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
	
	inline float operator()(float x, float y, float z)
	{
		return (*fofx)(x) + (*fofy)(y) + (*fofz)(z);
	}
};

#endif
