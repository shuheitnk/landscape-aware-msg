
function Y = msg_wrapper_batch(X, multi_msg, device)

    persistent torch_py device_py
    if isempty(torch_py)
        torch_py = py.importlib.import_module('torch');
    end
    if isempty(device_py) || ~strcmp(char(device_py.type), device)
        device_py = torch_py.device(device);
    end
    X = double(X);   % (N, D)
    x_tensor = torch_py.tensor( ...
        X, ...
        pyargs('dtype', torch_py.float32, 'device', device_py) ...
    );
    y_tensor = multi_msg(x_tensor);
    Y = double(y_tensor.detach().cpu().numpy());
    
end