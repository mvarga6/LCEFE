#include "experiment.h"

///
/// Experiment Class
///

bool Experiment::AddComponent(string name, ExperimentComponent* component)
{
    if (KeyExists(name))
    {
        return false;
    }

    components[name] = component;
    return true;
}

bool Experiment::Update(real dt)
{
    bool success = true;
    for (auto comp : components)
    {
        success = comp.second->Update(dt) && success;
    }
    return success;
}

bool Experiment::KeyExists(string key)
{
    return (components.find(key) != components.end());
}