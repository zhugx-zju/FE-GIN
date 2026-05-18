function [Bgauss,w,detJgauss,Dgauss,Efield] = getFGMBD(meshInfo,materialInfo)
	%%%===========================Description================================%%%
	%%% This function is to get the FGM element and stiffness matrix of FGM 
	%%% at Gauss Point based on Isoparametric Elements.
	%%%======================================================================%%%
	%%%======================================================================%%%

    [N,dNs,dNt,w] = shapeFunAtGauss();
	nEle = meshInfo.nEl;
	xEle = meshInfo.xEle;
    yEle = meshInfo.yEle;
    nodesx = meshInfo.nodesx;
    nodesy = meshInfo.nodesy;
    coord = meshInfo.coord;
    %% Assign Modulus Value for each element & Nodes
    nu = materialInfo.nu;
    X_mesh = reshape(coord(:,1),nodesx,nodesy);
    Y_mesh = reshape(coord(:,2),nodesx,nodesy);

    switch materialInfo.type
        case 'bil'
            alpha = materialInfo.alpha;
            beta = materialInfo.beta;
            GEle = 1.0 + alpha * xEle + beta * yEle;
            Efield = 1.0 + alpha * X_mesh' + beta * Y_mesh';
        case 'exp'
            alpha = materialInfo.alpha;
            beta = materialInfo.beta;
            GEle = exp(alpha * xEle + beta * yEle);
            Efield = exp(alpha * X_mesh' + beta * Y_mesh');
        case 'grf'
            % Use provided nodal E_field directly
            Efield = materialInfo.E_field;
            % Extract element nodal values using IGFE approach
            % Efield is [nodesy x nodesx], need to extract for each element
            Efield_vec = Efield';  % Transpose to match coord ordering
            Efield_vec = Efield_vec(:);  % Flatten to vector
            % Get element nodal values [nEle x 4]
            eleNodsID = meshInfo.eleNodsID;
            GEle = Efield_vec(eleNodsID);
        otherwise
            error('Invalid Distribution Type! Supported: ''bil'', ''exp'', ''grf''.');
    end
    % Get FGM element
	[nIntPoint,nShapeFun] = size(N);
	Bgauss = []; detJgauss =[];
	Dgauss = [];
	D_int = [1 , nu , 0;
		 nu, 1  , 0;
		 0 , 0  , 0.5*(1-nu)]/(1-nu^2);
	for i = 1:nIntPoint
		dNs_i = dNs(i,:); dNt_i = dNt(i,:);
		%% - J at gauss_i
		dxds = dNs_i*xEle'; dxdt = dNt_i*xEle';
		dyds = dNs_i*yEle'; dydt = dNt_i*yEle'; %size:(1*Nele)
		detJ = dxds.*dydt-dyds.*dxdt;
		dxds = dxds./detJ; dxdt = dxdt./detJ;
		dyds = dyds./detJ; dydt = dydt./detJ;
		%% - grad(u) & grad(v) in s-t coord-
		% dN*a; with a={u1,v2,...un,vn}
		duds = zeros(1,nShapeFun*2); dudt = duds;
		dvds = duds; dvdt = dvds;
		duds(1,1:2:end) = dNs_i; dudt(1,1:2:end) = dNt_i;
		dvds(1,2:2:end) = dNs_i; dvdt(1,2:2:end) = dNt_i;
		%% - grad(u) & grad(v) in x-y coord-
		dudx = (dydt)'*duds-(dyds)'*dudt;
		dudy = -(dxdt)'*duds+(dxds)'*dudt;
		dvdx = (dydt)'*dvds-(dyds)'*dvdt;
		dvdy = -(dxdt)'*dvds+(dxds)'*dvdt;
		%% - get strain
		ex = dudx; ey = dvdy;
		exy = dudy+dvdx;
		%% - Get B at gauss_i
		B(1,:,:) = (ex)';
		B(2,:,:) = (ey)';
		B(3,:,:) = (exy)';
		%% - 3D mat B to cell B
		Bele = mat2cell(B,[3],[8],ones(1,nEle));
		Bele_gauss_i = (Bele(:));
		Bgauss = [Bgauss,Bele_gauss_i];
		detJ_gauss_i = mat2cell(detJ',ones(1,nEle),1);
		detJgauss = [detJgauss,detJ_gauss_i];
		 
		N_i = N(i,:);
		Eele_gauss_i = N_i*GEle';
		D = kron(Eele_gauss_i',D_int);
		Dele = mat2cell(D,3*ones(1,nEle),[3]);
		Dgauss = [Dgauss,Dele];
    end
end