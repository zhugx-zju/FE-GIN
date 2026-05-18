%% Test GRF Generation
% This script tests the GRF_Generate function
clc; clear all

fprintf('=== Testing GRF Generation ===\n\n');

%% Parameters
geoL = 9; geoH = 9;
nelx = 40; nely = 40;
nodesx = nelx + 1; nodesy = nely + 1;
num = 5;  % Small number for testing
E_max = 6.0;
sigma_g = 1.0;
ell = 20;  % Fixed correlation length (larger = smoother)
seed_max = 42;

%% Create mesh info
fprintf('Creating mesh...\n');
[meshInfo, contourShow] = Mesh_Info(geoL, geoH, nelx, nely);

%% Generate GRF samples
fprintf('\nGenerating %d GRF samples...\n', num);
fprintf('  E_max range: [0.1, %.2f]\n', E_max);
fprintf('  sigma_g = %.2f\n', sigma_g);
fprintf('  ell (fixed) = %.2f\n\n', ell);

[E_field, E_max_vec] = GRF_Generate(num, E_max, sigma_g, ell, ...
                                     meshInfo, seed_max);

%% Display results
fprintf('\nGeneration complete!\n');
fprintf('E_field size: [%d x %d x %d]\n', size(E_field));
fprintf('Correlation length (fixed): %.3f (ell/L = %.3f)\n', ell, ell/geoL);
fprintf('Sample E_max values:\n');
for i = 1:num
    fprintf('  Sample %d: E_max = %.3f\n', i, E_max_vec(i));
end

%% Statistics
fprintf('\nStatistics for each sample:\n');
for i = 1:num
    E_sample = squeeze(E_field(i, :, :));
    fprintf('  Sample %d: mean = %.3f, std = %.3f, min = %.3f, max = %.3f\n', ...
            i, mean(E_sample(:)), std(E_sample(:)), ...
            min(E_sample(:)), max(E_sample(:)));
end

%% Visualization
fprintf('\nGenerating visualizations...\n');
figure('Position', [100, 100, 1200, 800]);

x = linspace(0, geoL, nodesx);
y = linspace(0, geoH, nodesy);

for i = 1:num
    subplot(2, 3, i);
    E_sample = squeeze(E_field(i, :, :));

    imagesc(x, y, E_sample);
    colorbar;
    axis equal tight;
    title(sprintf('Sample %d\nE_{max}=%.2f', i, E_max_vec(i)));
    xlabel('x'); ylabel('y');
end

sgtitle('GRF Modulus Fields');

fprintf('\nTest complete!\n');
