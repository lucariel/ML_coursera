function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%


pos = find(y == 1); neg = find(y == 0)

plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2,'MarkerSize', 7);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);


%%markerstyle
 %   ‘+’	crosshair
 %   ‘o’	circle
 %   ‘*’	star
 %   ‘.’	point
 %   ‘x’	cross
 %   ‘s’	square
 %   ‘d’	diamond
 %   ‘^’	upward-facing triangle
 %   ‘v’	downward-facing triangle
 %   ‘>’	right-facing triangle
 %   ‘<’	left-facing triangle
 %   ‘p’	pentagram
 %   ‘h’	hexagram

%Multiple property-value pairs may be specified, but they must appear in pairs. 
%These arguments are applied to the line objects drawn by plot. 
%Useful properties to modify are "linestyle", "linewidth", "color", "marker",
% "markersize", "markeredgecolor", "markerfacecolor".



% =========================================================================



hold off;

end
