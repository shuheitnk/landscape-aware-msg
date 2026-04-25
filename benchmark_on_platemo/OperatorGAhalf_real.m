function Offspring = OperatorGAhalf_real(Problem, Parent, Parameter)

    %% Generate full offspring
    if nargin > 2
        Offspring_full = OperatorGA(Problem, Parent, Parameter);
    else
        Offspring_full = OperatorGA(Problem, Parent);
    end

    %% Select half
    total = length(Offspring_full);  % SOLUTION対応
    idx   = randperm(total, floor(total/2));

    Offspring = Offspring_full(idx);

end