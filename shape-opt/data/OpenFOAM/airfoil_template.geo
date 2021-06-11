POINTS

Spline(1000) = {1000:LAST_POINT_INDEX,1000};

edge_lc = 0.8;
Point(19900) = { 25, 25, 0, edge_lc};
Point(19901) = { 25, -25, 0, edge_lc};
Point(19902) = { -25, -25, 0, edge_lc};
Point(19903) = { -25, 25, 0, edge_lc};

Line(1) = {19900,19901};
Line(2) = {19901,19902};
Line(3) = {19902,19903};
Line(4) = {19903,19900};

Line Loop (1) = {1,2,3,4};
Line Loop (2) = {1000};
Plane Surface(1) = {1,2};

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

