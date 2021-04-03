clear all
clc
viz = Visualizer2D;
viz.showTrajectory = false;
load exampleMap
p = zeros(40,50);
p(1,:) = ones(1, size(p,2));
p(size(p,1),:) = ones(1, size(p,2));
p(:,1) = ones(size(p,1),1);
p(:,size(p,2)) = ones(size(p,1),1);
%p(:,13) = [1; 0; 0; 0; 0; 0; 0; ones(20,1)];
objects = [20 25; 20 24; 20 23; 20 22; 20 21; 20 20; 20 19];
%objects = [1 2; 1 39];
loc = zeros(size(p,2), size(p,1));
loc(1,:) = ones(1, size(loc,2));
loc(size(loc,1),:) = ones(1, size(loc,2));
loc(:,1) = ones(size(loc,1),1);
loc(:,size(loc,2)) = ones(size(loc,1),1);

for i = 1:1:size(objects,1)
    p(size(p,1) - objects(i,2) + 1, objects(i,1) + 1) = 1;
    loc(objects(i,1), objects(i,2)) = 1;
end

mapRooz = binaryOccupancyMap(p, 1);
viz.mapName = 'mapRooz';
pose = [3; 4; 0];
viz(pose)