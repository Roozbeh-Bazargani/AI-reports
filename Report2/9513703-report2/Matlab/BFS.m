function [path,parent] = BFS(loc,start_state, goal_state)
% Actions = {N, S, E, W, NW, NE, SW, SE} |A| = 8
actions = [0 1; 0 -1; 1 0; -1 0; -1 1; 1 1; -1 -1; 1 -1]; % direction of Actions in x-y coordination
numActions = size(actions,1);
m = size(loc,1);
n = size(loc,2);
s = start_state;
parent = zeros(m, n);
visited = zeros(m, n);
queue = zeros(m*n, 2);
f = ones(m, n)*m*n;
q_head = 1;
queue(q_head, :) = s;
f(s(1),s(2)) = 0;
q_tail = 2;
while q_head ~= q_tail
    is_complete = false;
    s = queue(q_head, :);
    q_head = q_head + 1;
    for a = 1:numActions
        sp = s + actions(a, :);
        if sp(1) < 1 || sp(1) > size(loc,1) || sp(2) < 1 || sp(2) > size(loc,2) ...
                || visited(sp(1), sp(2)) == 1 || loc(sp(1), sp(2)) == 1
            continue;
        end
        visited(sp(1), sp(2)) = 1;
        f(sp(1), sp(2)) = f(s(1), s(2)) + norm(actions(a,:));
        parent(sp(1), sp(2)) = (s(2)-1)*m + s(1);
        queue(q_tail, :) = sp;
        q_tail = q_tail + 1;
        if s(1) == goal_state(1) && s(2) == goal_state(2)
            is_complete = true;
            break
        end
    end
    if is_complete
        break
    end
end

% constructing path
path = zeros(m*n, 2);
s = goal_state;
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

