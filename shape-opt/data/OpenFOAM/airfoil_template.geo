POINTS

Spline(1000) = {1000:LAST_POINT_INDEX,1000};
//BSpline(1000) = {1000:LAST_POINT_INDEX,1000};

edge_lc = 1.6;
Point(19900) = { 50, 50, 0, edge_lc};
Point(19901) = { 50, -50, 0, edge_lc};
Point(19902) = { -50, -50, 0, edge_lc};
Point(19903) = { -50, 50, 0, edge_lc};

Line(1) = {19900,19901};
Line(2) = {19901,19902};
Line(3) = {19902,19903};
Line(4) = {19903,19900};

Line Loop (1) = {1,2,3,4};
Line Loop (2) = {1000};
Plane Surface(1) = {1,2};


//extrude the boundary of the foil inwards by 0.05, with 5 layers of elements
//Extrude { Surface{234}; Layers{5, 0.05}; }
// NOTE : Create Boundaries inside the airfoil, looks not efficient for my case

//Define Boundary Layer
Field[1] = BoundaryLayer;
Field[1].EdgesList = {1000};


Field[1].hwall_n = 5e-3; //0.0001;
Field[1].thickness = 2.5e-2; //5e-3; //0.001;
Field[1].ratio = 1.05;
Field[1].AnisoMax = 5;
//Field[1].Quads = 10;
Field[1].IntersectMetrics = 10;
BoundaryLayer Field = 1;

Recombine Surface{1};

Extrude {0, 0, 0.1} {
  Surface{1};
  Layers{1};
  Recombine;
}

Physical Surface("back") = {1027};
Physical Surface("front") = {1};
Physical Surface("top") = {1022};
Physical Surface("exit") = {1010};
Physical Surface("bottom") = {1014};
Physical Surface("inlet") = {1018};
Physical Surface("aerofoil") = {1026};
Physical Volume("internal") = {1};

