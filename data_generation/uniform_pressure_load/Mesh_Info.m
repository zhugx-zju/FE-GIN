function [meshInfo,contourShow] = Mesh_Info(l,h,nelx,nely)
	%%%===========================Description================================%%%
	%%% This is a  function to mesh a rectangular region with uniform element
	%%% ----INPUT
	%%% l: <scalar> = length of the region
	%%% h: <scalar> = height of the region
	%%% nelx: <scalar> = number of elements in length direction
	%%% nely: <scalar> = number of elements in height direction
	%%% elementType: <String> = element Type (Only 'CPS4' is avaliable now)
	%%%
	%%% ----OUTPUT
	%%% coor: <vector,[Num of Nodes , 2]> = coordinates of each node
	%%% eleNodsID: <vector,[Num of element , 4]> = node ID for each element
	%%% el & eh: <scalar> = element length & height
	%%% plot_x: <matrix> = meshed x-coordinates for visualisation
	%%% plot_y: <matrix> = meshed y-coordinates for visualisation
	%%%======================================================================%%%
	el = l/nelx;
	eh = h/nely;

	nodesx = nelx+1;
	nodesy = nely+1;

	x = linspace(0,l,nodesx);
	y = linspace(0,h,nodesy);

	[plot_x,plot_y] = meshgrid(x,y);
	coor(:,1) = reshape(plot_x',[],1);
	coor(:,2) = reshape(plot_y',[],1);
    X = coor(:,1); Y = coor(:,2);

	%Assemble relationship
	nEl = nelx*nely;
	meshNods = int32(reshape((1:nodesx*nodesy),nodesx,nodesy));
	% 每个单元左下角节点的第一个自由度
	cVec = reshape(2*meshNods(1:end-1,1:end-1)+1,nEl,1);
	% 定义每个单元四个节点的两个自由度并组装
	cMat = cVec+int32([-2,-1,0,1,2*nelx+[2,3,0,1]]);
	eleNodsID = reshape(meshNods(1:end-1,1:end-1),nEl,1)+int32([0,1,nelx+[2,1]]);
	nDof = nodesx*nodesy*2;
	si = []; sii = [];
	for j= 1:8
		si = cat(2,si,j:8);
		sii = cat(2,sii,repmat(j,1,8-j+1));
	end
	ik = (cMat(:,si))'; jk = (cMat(:,sii))';
	Ivar = sort([ik(:),jk(:)],2,'descend');


    meshInfo.nodesx = nodesx; meshInfo.nodesy = nodesy;
	meshInfo.coord = coor; meshInfo.eleNodsID = eleNodsID;
	meshInfo.eleSize=[el,eh];meshInfo.eleDofID = cMat; meshInfo.Ivar = Ivar;
	meshInfo.nEl = nEl; meshInfo.el = el;  meshInfo.eh = eh; meshInfo.nNod = length(coor(:,1));
    meshInfo.nDof = nDof;
    meshInfo.xEle = X(eleNodsID); meshInfo.yEle = Y(eleNodsID);
    meshInfo.nelx = nelx; meshInfo.nely = nely;
    meshInfo.geo = [l,h];
	contourShow.PlotX = plot_x; contourShow.PlotY = plot_y;
	contourShow.elPlotSize = [nely,nelx];




