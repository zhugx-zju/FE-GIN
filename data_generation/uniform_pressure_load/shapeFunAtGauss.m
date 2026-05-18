function [N,dNs,dNt,w] = shapeFunAtGauss()
	%%%===========================Description================================%%%
	%%% Get the value of shape function at Gauss point
	%%%======================================================================%%%
    %%% 4 <- 3
    %%%      ^
    %%% 1 -> 2
    s = [-1/sqrt(3);1/sqrt(3);1/sqrt(3);-1/sqrt(3)];
	t = [-1/sqrt(3);-1/sqrt(3);1/sqrt(3);1/sqrt(3)];
	w = [1.0;1.0;1.0;1.0];
	%shape fun
	N1 = (1-s).*(1-t)/4;
    N2 = (1+s).*(1-t)/4;
    N3 = (1+s).*(1+t)/4;
	N4 = (1-s).*(1+t)/4;

	N1s = -(1-t)/4; N1t = -(1-s)/4;
    N2s = (1-t)/4;  N2t = -(1+s)/4;
	N4s = -(1+t)/4; N4t = (1-s)/4;
    N3s = (1+t)/4;  N3t = (1+s)/4;

    %cal N
	N = [N1,N2,N3,N4];
	dNs = [N1s,N2s,N3s,N4s];
	dNt = [N1t,N2t,N3t,N4t];
end