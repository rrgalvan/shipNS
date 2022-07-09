SetFactory("OpenCASCADE");

Mesh.CharacteristicLengthMax = 2.5;
Mesh.CharacteristicLengthMin = 0.8;

// Rectángulo con esquina inf-izq (0, -0.5, 0) de 3x1
Rectangle(1)    = {0., -20, 0,  120, 40};

// Elipses paralelas
Disk(2) = { 50, -3.2, 0., 7.2, 0.8};
Disk(3) = { 50, +3.2, 0., 7.2, 0.8};

BooleanDifference{ Surface{1}; Delete;}{ Surface{2}; Surface{3}; Delete;}
