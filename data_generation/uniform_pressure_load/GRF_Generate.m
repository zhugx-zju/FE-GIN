function [E_field, E_max_vec] = GRF_Generate(num, E_max, sigma_g, ell, meshInfo, seed_max)
    % Generate Gaussian Random Field (GRF) samples with RBF kernel (Optimized)
    % Similar to GFE_Generate, generates E_max vector for each sample
    % All samples use the same correlation length for smooth fields
    %
    % Inputs:
    %   num         - Number of samples to generate
    %   E_max       - Maximum value for E_max (e.g., 8.0)
    %   sigma_g     - Standard deviation of GRF (controls variability, e.g., 1.0)
    %   ell         - Correlation length (fixed for all samples, larger = smoother)
    %   meshInfo    - Mesh information structure from Mesh_Info
    %   seed_max    - Random seed for E_max shuffle
    %
    % Outputs:
    %   E_field    - Generated modulus fields [num x nodesy x nodesx]
    %   E_max_vec  - Maximum modulus for each sample [num x 1]
    %
    % Method (similar to GFE_Generate):
    %   1. Generate E_max vector using linspace (0.1 to E_max)
    %   2. Shuffle with seed for reproducibility
    %   3. Generate GRF for each sample with range [0, E_max(i)]
    %   4. Use fixed correlation length for all samples (smooth fields)

    % Generate E_max values using linspace (similar to Ex/Ey in GFE_Generate)
    E_max_vec = linspace(1.0, E_max, num)';

    % Shuffle with seed (similar to GFE_Generate)
    rng(seed_max);
    E_max_vec = E_max_vec(randperm(num));

    % Extract coordinates from meshInfo
    coords = meshInfo.coord;
    n_points = size(coords, 1);
    nodesx = meshInfo.nodesx;
    nodesy = meshInfo.nodesy;

    % Pre-compute squared distance matrix
    fprintf('Pre-computing distance matrix...\n');
    D_sq = compute_distance_matrix_vectorized(coords);
    fprintf('  Distance matrix computed: [%d x %d]\n', size(D_sq));

    % Pre-compute RBF covariance matrix (same for all samples with fixed ell)
    fprintf('Pre-computing RBF covariance matrix (ell = %.3f)...\n', ell);
    K = exp(-D_sq / (2 * ell^2));
    K = K + 1e-6 * eye(n_points);

    % Cholesky decomposition (only once!)
    L = chol(K, 'lower');
    fprintf('  Cholesky decomposition complete\n');

    % Initialize output
    E_field = zeros(num, nodesy, nodesx);

    % Generate samples
    fprintf('Generating %d GRF samples...\n', num);
    tic;
    for i = 1:num
        if mod(i, 100) == 0
            elapsed = toc;
            avg_time = elapsed / i;
            remaining = avg_time * (num - i);
            fprintf('  Progress: %d/%d (%.1f%%, est. remaining: %.1fs)\n', ...
                    i, num, 100*i/num, remaining);
        end

        % Get E_max for this sample
        E_tmp = E_max_vec(i);

        % Generate GRF sample (using pre-computed L)
        z = randn(n_points, 1);
        g = L * z;

        % Reshape to grid
        g_grid = reshape(g, nodesx, nodesy)';

        % Normalize GRF to [0, E_max(i)]
        % Use tanh to map to [0, 1], then scale to [0, E_max]
        g_normalized = tanh(sigma_g * g_grid);  % Maps to [-1, 1]
        g_normalized = (g_normalized + 1) / 2;  % Maps to [0, 1]
        E_grid = E_tmp * g_normalized;          % Maps to [0, E_max(i)]

        % Store result
        E_field(i, :, :) = E_grid;
    end

    total_time = toc;
    fprintf('GRF generation complete! Total time: %.2fs (avg: %.3fs/sample)\n', ...
            total_time, total_time/num);
end

function D_sq = compute_distance_matrix_vectorized(coords)
    % Compute squared pairwise distance matrix using vectorized operations
    sq_norms = sum(coords.^2, 2);
    dot_products = coords * coords';
    D_sq = sq_norms + sq_norms' - 2 * dot_products;
    D_sq = max(D_sq, 0);
    D_sq = (D_sq + D_sq') / 2;
end
