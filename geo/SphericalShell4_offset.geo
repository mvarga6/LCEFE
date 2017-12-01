//+
SetFactory("OpenCASCADE");

Sphere(1) = {0, 0, 0, 30, -Pi/2, Pi/2, 2*Pi};
Sphere(2) = {1.5, 0, 0, 27, -Pi/2, Pi/2, 2*Pi};
BooleanDifference(3) = { Volume{1}; Delete; }{ Volume{2}; Delete; };

Mesh.Algorithm3D = 1;
Mesh.CharacteristicLengthMin = 1.0;
Mesh.CharacteristicLengthMax = 1.0;

Mesh 3;
Mesh.Format = 1;
Mesh.SaveAll = 1;
