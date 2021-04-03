function [path, parent] = Greedy_best_first_search(loc,start_state, goal_state)
% f(s) = h(s)
% h(s): estimated cost of the cheapest path from s to goal_state => 
% h(s) = norm(pos(s) - pos(goal))
% Actions = {N, S, E, W, NW, NE, SW, SE} |A| = 8
actions = [0 1; 0 -1; 1 0; -1 0; -1 1; 1 1; -1 -1; 1 -1]; % direction of Actions in x-y coordination
numActions = size(actions,1);
m = size(loc,1);
n = size(loc,2);
s = start_state;
parent = zeros(m, n);
f = ones(m, n)*m*n;
visited = zeros(m, n);
while ~(s(1) == goal_state(1) && s(2) == goal_state(2))
    visited(s(1), s(2)) = 1;
    f(s(1), s(2)) = m*n;
    % updating nodes
    for a = 1:numActions
        sp = s + actions(a,:);
        if sp(1) < 1 || sp(1) > size(loc,1) || sp(2) < 1 || sp(2) > size(loc,2) ...
                || visited(sp(1), sp(2)) == 1 || loc(sp(1), sp(2)) == 1
            continue
        end
        h = norm(goal_state - sp);
        if h < f(sp(1), sp(2))
            f(sp(1), sp(2)) = h;
            parent(sp(1), sp(2)) = (s(2)-1)*m + s(1);
        end
    end
    % searching
    M = min(f,[], 'all');
    [x, y] = find(f==M);
    s = [x(1) y(1)];
%     fprintf('s = (%d, %d): f = %d)\n', s(1), s(2), M);
end

% constructing path
path = zeros(m*n, 2);
path(m*n,:) = s;
i = m*n;
while ~(s(1) == start_state(1) && s(2) == start_state(2))
    sp = parent(s(1), s(2));
    s = [mod(sp,m) ceil(sp/m)];
    i = i - 1;
    path(i,:) = s;
    %fprintf('s = (%d, %d)\n', s(1), s(2))
end

[x, ~] = find(path==0);
path = path(x(size(x,1)) + 1:m*n,:);

end

