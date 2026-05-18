function [Ex, Ey, alpha, beta] = GFE_Generate(Batch_Num, dis_type, E1, E2, geoX, geoY, seedX, seedY)
    % Generate Ex, Ey arrays based on distribution type, return alpha and beta
    quart = Batch_Num / 4;  % 1/4 of total batches
    switch dis_type
        case 'bil'
            % X-Y- region: 0.51→1.0
            Ex1 = linspace(0.51, 1.0, quart);
            Ey1 = linspace(0.51, 1.0, quart);
            % X-Y+ region: Ex 0.51→1.0, Ey 1.0→E2
            Ex2 = linspace(0.51, 1.0, quart);
            Ey2 = linspace(1.0, E2, quart);
            % X+Y- region: Ex 1.0→E1, Ey 0.51→1.0
            Ex3 = linspace(1.0, E1, quart);
            Ey3 = linspace(0.51, 1.0, quart);
            % X+Y+ region: Ex 1.0→E1, Ey 1.0→E2
            Ex4 = linspace(1.0, E1, quart);
            Ey4 = linspace(1.0, E2, quart);

            Ex = [Ex1, Ex2, Ex3, Ex4];
            Ey = [Ey1, Ey2, Ey3, Ey4];

            % Shuffle with seeds
            rng(seedX);
            Ex = Ex(randperm(length(Ex)));
            rng(seedY);
            Ey = Ey(randperm(length(Ey)));

            % Linear parameters
            alpha = (Ex - 1.0) / geoX;
            beta = (Ey - 1.0) / geoY;

        case 'exp'
            % X-Y- region: 0.1→1.0
            Ex1 = linspace(0.1, 1.0, quart);
            Ey1 = linspace(0.1, 1.0, quart);
            % X-Y+ region: Ex 0.1→1.0, Ey 1.0→E2
            Ex2 = linspace(0.1, 1.0, quart);
            Ey2 = linspace(1.0, E2, quart);
            % X+Y- region: Ex 1.0→E1, Ey 0.1→1.0
            Ex3 = linspace(1.0, E1, quart);
            Ey3 = linspace(0.1, 1.0, quart);
            % X+Y+ region: Ex 1.0→E1, Ey 1.0→E2
            Ex4 = linspace(1.0, E1, quart);
            Ey4 = linspace(1.0, E2, quart);

            Ex = [Ex1, Ex2, Ex3, Ex4];
            Ey = [Ey1, Ey2, Ey3, Ey4];

            % Shuffle with seeds
            rng(seedX);
            Ex = Ex(randperm(length(Ex)));
            rng(seedY);
            Ey = Ey(randperm(length(Ey)));

            % Exponential parameters
            alpha = log(Ex) / geoX;
            beta = log(Ey) / geoY;

        otherwise
            error('Invalid Distribution Type! Supported: ''2D Linear'', ''2D Exponential''.');
    end
end