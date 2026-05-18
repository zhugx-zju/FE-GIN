%% Simualte batches of FGMs under random pressure load
clc; clear all
%%% - Add function lab -
addpath(genpath('../stenglib-master'));
%% Geo & Mesh
geoL = 9; geoH = 9;
nelL = 40; nelH = 40;
nodesx = nelL+1;
nodesy = nelH+1;
[meshInfo,contourShow] = Mesh_Info(geoL,geoH,nelL,nelH);
%% Material Information
num = 20000;
nu = 0.3; % Poisson's ratio
dis_type = 'bil'; % 'bil' 'exp'
materialInfo.type = dis_type;
materialInfo.nu = nu;
%%% X Direction Range 
Ex_max = 3;
seed_X = 2; % 2 for 2d linear    4 for 2d exponential
%%% Y Direction Range
Ey_max = 3;
seed_Y = 3; % 3 for 2d linear    6 for 2d exponential
%%% Random modulus coefficients in X & Y directions
[Ex, Ey, alpha, beta] = GFE_Generate( ...
    num, dis_type, Ex_max, Ey_max, geoL, geoH, seed_X, seed_Y);
%% Displacement Boundary Condition
coord = meshInfo.coord; eleSize= meshInfo.eleSize; numNod = meshInfo.nNod;
fixEdge = find(coord(:,1)==0); fixdofs = sort([2*fixEdge-1;2*fixEdge(1)]);
loadEdge = find(coord(:,1)==geoL); loaddofs = loadEdge*2-1;
alldof = meshInfo.nDof; force = sparse(alldof,1); 
U_tmp = zeros(alldof,1);
%%% Random pressure load
F_tot = 0.01 * ones(1,num);
F_density = F_tot / geoH;
FMag = F_density*meshInfo.eh; 
%% BC Assemble
BCInfo.fixdof = fixdofs;
BCInfo.loaddof = loaddofs;
%% Batch Simulations
% Data Set
U = zeros(num,2,nodesy,nodesx);
E = zeros(num,nodesy,nodesx);
F = zeros(num,meshInfo.nDof,1);
strain_request = false;
BCInfo.strain_request = strain_request;
% Save directly to unified data directory
if strcmp(dis_type, 'exp')
    data_dir = "../../data/data_exp/force_load";
elseif strcmp(dis_type, 'bil')
    data_dir = "../../data/data_bil/force_load";
end
if ~exist(data_dir, 'dir')
    mkdir(data_dir);
end
if strain_request
    strain = zeros(num,3,nodesy,nodesx);
    for i = 1:num
        % Material Property
        materialInfo.alpha = alpha(i);
        materialInfo.beta = beta(i);
        % pressure Load
        force(loaddofs,1) = FMag(i);
        force(loaddofs(1),1) = force(loaddofs(1),1) - FMag(i) / 2;
        force(loaddofs(end),1) = force(loaddofs(end),1) - FMag(i) / 2;
        BCInfo.U = U_tmp;
        BCInfo.force = force;
        femInfo = FEM_Assemble(meshInfo,materialInfo,BCInfo);
        E_tmp = femInfo.Efield;
        Plot_Surf(contourShow,E_tmp);
        E(i,:,:) = E_tmp;
        % Solve IGFE
        state = Forward_Solver(femInfo);
        U(i,1,:,:) = state.ux;
        U(i,2,:,:) = state.uy;
        strain(i,1,:,:) = state.exx;
        Plot_Surf(contourShow,state.exx);
        strain(i,2,:,:) = state.eyy;
        strain(i,3,:,:) = state.exy;
        F(i,:,1) = state.RF;
    end
    % Save with simplified names
    save(fullfile(data_dir, "input.mat"), "U");
    save(fullfile(data_dir, "output.mat"), "E");
    save(fullfile(data_dir, "strain.mat"), "strain");
    save(fullfile(data_dir, "force.mat"), "F");
    save(fullfile(data_dir, "alpha.mat"), "alpha");
    save(fullfile(data_dir, "beta.mat"), "beta");
    fprintf('Data saved to: %s\n', data_dir);
else
    for i = 1:num
        % Material Property
        materialInfo.alpha = alpha(i);
        materialInfo.beta = beta(i);
        % Pressure Load
        force(loaddofs,1) = FMag(i);
        force(loaddofs(1),1) = force(loaddofs(1),1) - FMag(i) / 2;
        force(loaddofs(end),1) = force(loaddofs(end),1) - FMag(i) / 2;
        BCInfo.force = force;
        BCInfo.U = U_tmp;
        femInfo = FEM_Assemble(meshInfo,materialInfo,BCInfo);
        E_tmp = femInfo.Efield;
        Plot_Surf(contourShow,E_tmp);
        E(i,:,:) = E_tmp;
        % Solve IGFE
        state = Forward_Solver(femInfo);
        U(i,1,:,:) = state.ux;
        U(i,2,:,:) = state.uy;
        F(i,:,1) = state.RF;
    end
    % Save with simplified names
    save(fullfile(data_dir, "input.mat"), "U");
    save(fullfile(data_dir, "output.mat"), "E");
    save(fullfile(data_dir, "force.mat"), "F");
    save(fullfile(data_dir, "alpha.mat"), "alpha");
    save(fullfile(data_dir, "beta.mat"), "beta");
    fprintf('Data saved to: %s\n', data_dir);
end

