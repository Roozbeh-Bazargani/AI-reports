function [A_star_time, Greedy_best_time, BFS_time] = Heuristic_time(loc, iterations)
A_star_time = 0;
Greedy_best_time = 0;
BFS_time = 0;
for i=1:iterations
    start_state = [randi(size(loc,1)-2)+1 randi(size(loc,2)-2)+1];
    goal_state = [randi(size(loc,1)-2)+1 randi(size(loc,2)-2)+1];
    while loc(start_state(1), start_state(2)) == 1
        start_state = [randi(size(loc,1)-2)+1 randi(size(loc,2)-2)+1];
    end
    while loc(goal_state(1), goal_state(2)) == 1
        goal_state = [randi(size(loc,1)-2)+1 randi(size(loc,2)-2)+1];
    end
    tic %pair 1
    A_star(loc, start_state, goal_state);
    A_star_time = A_star_time + toc; % pair 1
    
    tic %pair 2
    Greedy_best_first_search(loc, start_state, goal_state);
    Greedy_best_time = Greedy_best_time + toc; % pair 2
    
    tic %pair 3
    BFS(loc, start_state, goal_state);
    BFS_time = BFS_time + toc; % pair 3
end
A_star_time = A_star_time / iterations;
Greedy_best_time = Greedy_best_time / iterations;
BFS_time = BFS_time / iterations;

end

