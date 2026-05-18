function [N,dNds,dNdt] = shapeFunAtNode(nEl)
	%%%===========================Description================================%%%
	%%% Get the value of shape function at Gauss point
	%%%======================================================================%%%
    %%% 4 <- 3
    %%%      ^
    %%% 1 -> 2
    s = [-1;1;1;-1];
	t = [-1;-1;1;1];
	% value
	N1 = repmat((1-s)'.*(1-t)'/4,nEl,1);
    N2 = repmat((1+s)'.*(1-t)'/4,nEl,1);
    N3 = repmat((1+s)'.*(1+t)'/4,nEl,1);
	N4 = repmat((1-s)'.*(1+t)'/4,nEl,1);
    N = {N1,N2,N3,N4};
    % grad
	N1ds = repmat(-(1-t)'/4,nEl,1); N1dt = repmat(-(1-s)'/4,nEl,1);
    N2ds = repmat((1-t)'/4,nEl,1);  N2dt = repmat(-(1+s)'/4,nEl,1);
	N4ds = repmat(-(1+t)'/4,nEl,1); N4dt = repmat((1-s)'/4,nEl,1);
    N3ds = repmat((1+t)'/4,nEl,1);  N3dt = repmat((1+s)'/4,nEl,1);
	dNds = {N1ds,N2ds,N3ds,N4ds};
	dNdt = {N1dt,N2dt,N3dt,N4dt};
end