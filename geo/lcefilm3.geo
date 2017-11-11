Point(1) = {0, 0, 0, 1.0};
Point(2) = {0.6, 0.3, 0.2, 1.0};
Point(3) = {0.3, 0.1, 0, 1.0};
Circle(1) = {1, 3, 2};
Delete {
  Line{1};
}
Delete {
  Point{2};
}
Delete {
  Point{1};
}
Delete {
  Point{3};
}
Point(3) = {0, 0, 0, 1.0};
Point(4) = {5, 0, 0, 1.0};
Point(5) = {-5, 0, 0, 1.0};
Circle(1) = {5, 3, 4};
Extrude {{1, 0, 0}, {-5, 0, 0}, 2*Pi} {
  Line{1};
}
Extrude {{1, 0, 0}, {-5, 0, 0}, Pi} {
  Line{1};
}
Extrude {{1, 0, 0}, {0, 0, 0}, Pi} {
  Line{1};
}
Extrude {{0, 1, 0}, {0, 0, 0}, Pi} {
  Line{1};
}
