%% Generate batches of GRF samples under uniform pressure load
clc; clear all

%%% - Add function lab -
addpath(genpath('../stenglib-master'));

%% Geo & Mesh
geoL = 9; geoH = 9;
nelL = 40; nelH = 40;
nodesx = nelL + 1;
nodesy = nelH + 1;
[meshInfo, contourShow] = Mesh_Info(geoL, geoH, nelL, nelH);

%% Material Information
num = 20000;
nu = 0.3; % Poisson's ratio
materialInfo.nu = nu;

%% GRF Parameters
E_max = 8.0;        % Maximum value for E_max
sigma_g = 1.0;      % GRF variability (controls field variation)
ell = 25;          % Correlation length (fixed, larger = smoother fields)
seed_max = 4;       % Random seed for E_max shuffle

fprintf('=== GRF Dataset Generation ===\n');
fprintf('Number of samples: %d\n', num);
fprintf('E_max range: [0.1, %.2f]\n', E_max);
fprintf('GRF variability sigma_g: %.2f\n', sigma_g);
fprintf('Correlation length (fixed): %.2f\n', ell);

%% Generate GRF modulus fields
fprintf('\nGenerating GRF modulus fields...\n');
[E_field, E_max_vec] = GRF_Generate(num, E_max, sigma_g, ell, ...
                                     meshInfo, seed_max);

%% Displacement Boundary Condition
coord = meshInfo.coord;
eleSize = meshInfo.eleSize;
numNod = meshInfo.nNod;

% Fixed edge (left boundary)
fixEdge = find(coord(:,1) == 0);
fixdofs = sort([2*fixEdge-1; 2*fixEdge(1)]);

% Load edge (right boundary)
loadEdge = find(coord(:,1) == geoL);
loaddofs = loadEdge * 2 - 1;

alldof = meshInfo.nDof;
force = sparse(alldof, 1);
U_tmp = zeros(alldof, 1);

%%% Random pressure load
F_tot = 0.01 * ones(1, num);
F_density = F_tot / geoH;
FMag = F_density * meshInfo.eh;

%% BC Assemble
BCInfo.fixdof = fixdofs;
BCInfo.loaddof = loaddofs;

%% Batch Simulations
fprintf('\nRunning forward simulations...\n');

% Data Set
U = zeros(num, 2, nodesy, nodesx);
E = zeros(num, nodesy, nodesx);
F = zeros(num, meshInfo.nDof, 1);

strain_request = false;
BCInfo.strain_request = strain_request;

% Create output directory
% Save directly to unified data directory
data_dir = "../../data/data_grf/force_load";
if ~exist(data_dir, 'dir')
    mkdir(data_dir);
end

if strain_request
    strain = zeros(num, 3, nodesy, nodesx);
    for i = 1:num
        if mod(i, 100) == 0
            fprintf('  Progress: %d/%d\n', i, num);
        end

        % Get GRF modulus field
        E_tmp = squeeze(E_field(i, :, :));

        % Material property (use nodal modulus directly)
        materialInfo.type = 'grf';
        materialInfo.E_field = E_tmp;

        % Pressure Load
        force(loaddofs, 1) = FMag(i);
        force(loaddofs(1), 1) = force(loaddofs(1), 1) - FMag(i) / 2;
        force(loaddofs(end), 1) = force(loaddofs(end), 1) - FMag(i) / 2;

        BCInfo.U = U_tmp;
        BCInfo.force = force;

        % Assemble FEM (getFGMBD now handles GRF type)
        femInfo = FEM_Assemble(meshInfo, materialInfo, BCInfo);

        E(i, :, :) = E_tmp;

        % Solve IGFE
        state = Forward_Solver(femInfo);
        U(i, 1, :, :) = state.ux;
        U(i, 2, :, :) = state.uy;
        strain(i, 1, :, :) = state.exx;
        strain(i, 2, :, :) = state.eyy;
        strain(i, 3, :, :) = state.exy;
        F(i, :, 1) = state.RF;
    end

    % Save data
    % Save with simplified names (no suffix)
    save(fullfile(data_dir, "input.mat"), "U");
    save(fullfile(data_dir, "output.mat"), "E");
    save(fullfile(data_dir, "strain.mat"), "strain");
    save(fullfile(data_dir, "force.mat"), "F");
    save(fullfile(data_dir, "ell.mat"), "ell_values");
    fprintf('Data saved to: %s\n', data_dir);
else
    for i = 1:num
        if mod(i, 100) == 0
            fprintf('  Progress: %d/%d\n', i, num);
        end

        % Get GRF modulus field
        E_tmp = squeeze(E_field(i, :, :));

        % Material property (use nodal modulus directly)
        materialInfo.type = 'grf';
        materialInfo.E_field = E_tmp;

        % Pressure Load
        force(loaddofs, 1) = FMag(i);
        force(loaddofs(1), 1) = force(loaddofs(1), 1) - FMag(i) / 2;
        force(loaddofs(end), 1) = force(loaddofs(end), 1) - FMag(i) / 2;

        BCInfo.force = force;
        BCInfo.U = U_tmp;

        % Assemble FEM (getFGMBD now handles GRF type)
        femInfo = FEM_Assemble(meshInfo, materialInfo, BCInfo);

        E(i, :, :) = E_tmp;

        % Solve IGFE
        state = Forward_Solver(femInfo);
        U(i, 1, :, :) = state.ux;
        U(i, 2, :, :) = state.uy;
        F(i, :, 1) = state.RF;
    end

    % Save data
    % Save with simplified names (no suffix)
    save(fullfile(data_dir, "input.mat"), "U");
    save(fullfile(data_dir, "output.mat"), "E");
    save(fullfile(data_dir, "force.mat"), "F");
    save(fullfile(data_dir, "E_max.mat"), "E_max_vec");
    fprintf('Data saved to: %s\n', data_dir);
end

fprintf('\n=== GRF Dataset Generation Complete ===\n');
fprintf('Data saved to: %s\n', data_dir);
