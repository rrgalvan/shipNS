SetFactory("OpenCASCADE");

Mesh.CharacteristicLengthMax = 0.07;
Mesh.CharacteristicLengthMin = 0.07;

// Rectángulo con esquina inf-izq (0,0,0) de 3x1
Rectangle(1)    = {0., 0., 0,  3, 1};

// Disco de radio 0.2 centrado en (1,0.5,0)
Disk(2) = { 1., 0.5,  0.,  0.1};

BooleanDifference{ Surface{1}; Delete;}{ Surface{2}; Delete;}
