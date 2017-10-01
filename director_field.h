#ifndef __DIRECTOR_FIELD_H__
#define __DIRECTOR_FIELD_H__

struct DirectorOrientation
{
	float theta, phi;
};

class DirectorField
{
public:
	virtual DirectorOrientation GetDirectorAt(const float, const float, const float) = 0;	
};

class UniformField : public DirectorField
{
	DirectorOrientation const_director;
public:
	UniformField(const float theta, const float phi);
	DirectorOrientation GetDirectorAt(const float x, const float y, const float z);
};

#endif
