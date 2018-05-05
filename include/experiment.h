#ifndef __EXPERIMENT_H__
#define __EXPERIMENT_H__

#include <map>
#include <string>
#include <vector>

#include "pointer.h"
#include "defines.h"

using namespace std;

///
/// Abstract parent of any aspect of an experiment
/// Examples: 
///     Update order parameter globally.
///     Update temperature.
///     Calculate external forces. 
///
class ExperimentComponent
{
public:
    bool virtual Update(real dt) = 0;

protected:
    bool virtual UpdateCpu(real dt) = 0;
    bool virtual UpdateGpu(real dt) = 0;
};

///
/// Changes the orger parameter for all tetrahedra by a set amount
///
class NematicToIsotropic : public ExperimentComponent
{
    real t; // stores aggregated time
    real dsdt, t_start, t_stop; // settings
    PointerHandle<real> s; // handle for data to manipulate
public:
    NematicToIsotropic(real tStart, real tStop, PointerHandle<real> S);
    
    bool Update(real dt);
    bool UpdateCpu(real dt);
    bool UpdateGpu(real dt);
};

class Clamp : public ExperimentComponent
{
    PointerHandle<real> f;
    std::vector<int> nodes;
public:
    Clamp(std::vector<int> nodeIdxList, PointerHandle<real> F);

    bool Update(real dt);
    bool UpdateCpu(real dt);
    bool UpdateGpu(real dt);
};

///
/// Groups together sets of experiment components
/// into one container so it can be passed to the simulation
/// runner.
///
class Experiment
{
    std::map<string, ExperimentComponent*> components;
public:
    bool AddComponent(string name, ExperimentComponent* component);
    bool Update(real dt);

private:
    bool KeyExists(string key);
};

#endif