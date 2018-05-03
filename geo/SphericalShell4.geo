//+
SetFactory("OpenCASCADE");

Sphere(1) = {0, 0, 0, 50, -Pi/2, Pi/2, 2*Pi};
Sphere(2) = {0, 0, 0, 46, -Pi/2, Pi/2, 2*Pi};
BooleanDifference(3) = { Volume{1}; Delete; }{ Volume{2}; Delete; };

Mesh.Algorithm3D = 1;
Mesh.CharacteristicLengthMin = 0.75;
Mesh.CharacteristicLengthMax = 1.25;

Mesh 3;
Mesh.Format = 1;
Mesh.SaveAll = 1;
