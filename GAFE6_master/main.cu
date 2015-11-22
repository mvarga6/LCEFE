//=============================================================//
//                                                             //
//            ||Gpu Accelerated Fineite Element ||             //
//                                                             //
//              --------Version 6.0----------                  //
//                                                             //
//                                                             //
//                                                             //
//    Authors: Andrew Konya      (Kent State University)       //
//             Robin Selinger    (Kent State University)       // 
//             Badel MBanga      (kent State University)       //
//                                                             //
//   Finite elemnt simulation executed on GPU using CUDA       //
//   Hybrid MD finite element algorithm used to allow          //
//   all computations be implemented locally requireing        //
//   parallelization of all prccess in calculation             //
//                                                             //
//=============================================================//


#include "mainhead.h"
#include "mesh.h"
#include "DeviceController.h"

int main()
{
	Mesh * mesh = new Mesh;
	mesh->loadMeshDim();
	mesh->createTetAndNodeArrays();
	mesh->loadMesh();
	mesh->rescaleMesh(2.0f, AXIS::X);
	mesh->rescaleMesh(0.25f, AXIS::Y);
	mesh->rescaleMesh(2.0f, AXIS::Z);
	mesh->calculateTetPositions();
	mesh->loadDirector();
	mesh->orderTetAndNodeArrays();
	mesh->calculateAMatrices();
	mesh->printOrderAndDirector();

	DeviceController * gpu_controller = new DeviceController(mesh);
	gpu_controller->packData();
	gpu_controller->dataHostToDevice();
	gpu_controller->runDynamics();

    return 0;
}
