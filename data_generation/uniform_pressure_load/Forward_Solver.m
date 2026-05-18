function state = Forward_Solver(femInfo)
	%%%===========================Description================================%%%
	%%% This is a  function to get displacement results under displacement load
    %%% condition.
	%%%======================================================================%%%
    % Assemble Total Stiffness Matrix
    meshInfo = femInfo.meshInfo;
    nDof = meshInfo.nDof;
    Ivar = meshInfo.Ivar;
    nodesx = meshInfo.nodesx;
    nodesy = meshInfo.nodesy;
    keVec = cell2mat(femInfo.keVec);
    K = fsparse(Ivar(:,1),Ivar(:,2),keVec,[nDof,nDof]);
    % Load vector
    BCInfo = femInfo.BCInfo;
    force = BCInfo.force;
    fixdof = BCInfo.fixdof;
    U = BCInfo.U;
    % FEM Solve
    freedofs = setdiff(1:nDof,fixdof);
    U(freedofs,:) = decomposition(K(freedofs,freedofs),'chol','lower') \ force(freedofs,:);
    % Save data
    ux = reshape(U(1:2:end)',[],1);
    uy = reshape(U(2:2:end)',[],1);
    K = K + K' - diag(diag(K));
    state.U = U;
    state.RF = K*U;
    state.ux = reshape(state.U(1:2:end),nodesx,nodesy)';
    state.uy = reshape(state.U(2:2:end),nodesx,nodesy)';
    % Calculate node strain
    if BCInfo.strain_request
        [~,dNds,dNdt] = shapeFunAtNode(meshInfo.nEl);
        [locJ,detJ] = Local_InvJ(meshInfo,dNds,dNdt);
        [exx,eyy,exy] = Dis2Strain(meshInfo,locJ,detJ,dNds,dNdt,ux,uy);
        [node_exx,node_eyy,node_exy] = GetNodeStrain(meshInfo,exx,eyy,exy);        
        state.exx = node_exx;
        state.eyy = node_eyy;
        state.exy = node_exy;
    end
end