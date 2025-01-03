classdef SMLP
    properties
        n_inputs 
        n_outputs 
        n_hidden_layers 
        n_hidden_units 
        hidden_weights 
        output_weights 
        learning_rate 
        momentum 
        prev_hidden_weight_delta 
        prev_output_weight_delta 
    end
    
    methods
        %Construction
        function obj = SMLP(n_inputs, n_outputs, n_hidden_layers, n_hidden_units, learning_rate, momentum)
            obj.n_inputs = n_inputs;
            obj.n_outputs = n_outputs;
            obj.n_hidden_layers = n_hidden_layers;
            obj.n_hidden_units = n_hidden_units;
            obj.learning_rate = learning_rate;
            obj.momentum = momentum;
            
            
            obj.hidden_weights = cell(1, n_hidden_layers);
            obj.output_weights = randn(n_hidden_units(end), n_outputs);
            
            for i = 1:n_hidden_layers
                if i == 1
                    input_size = n_inputs;
                else
                    input_size = n_hidden_units(i-1);
                end
                obj.hidden_weights{i} = randn(input_size, n_hidden_units(i));
            end
            
           
            obj.prev_hidden_weight_delta = cell(1, n_hidden_layers);
            obj.prev_output_weight_delta = zeros(n_hidden_units(end), n_outputs);
            for i = 1:n_hidden_layers
                obj.prev_hidden_weight_delta{i} = zeros(size(obj.hidden_weights{i}));
            end
        end
        
        function hidden_outputs = encode(obj, input)
            
            hidden_outputs = cell(1, obj.n_hidden_layers);
            for i = 1:obj.n_hidden_layers
                if i == 1
                    hidden_input = input * obj.hidden_weights{i};
                else
                    hidden_input = hidden_outputs{i-1} * obj.hidden_weights{i};
                end
                hidden_outputs{i} = sigmoid(hidden_input);
            end
        end

        function output = decode(obj, hidden_outputs)
            
            output = sigmoid(hidden_outputs * obj.output_weights);
        end
        
        function train(obj, inputs, targets)
            
            [output, hidden_outputs] = obj.forward(inputs);
            
            
            output_error = (output - targets) .* sigmoid_derivative(output);
            hidden_errors = cell(1, obj.n_hidden_layers);
            hidden_errors{end} = output_error * obj.output_weights' .* sigmoid_derivative(hidden_outputs{end});
            for i = obj.n_hidden_layers-1:-1:1
                hidden_errors{i} = hidden_errors{i+1} * obj.hidden_weights{i+1}' .* sigmoid_derivative(hidden_outputs{i});
            end
            
            
            obj.prev_output_weight_delta = obj.momentum * obj.prev_output_weight_delta - obj.learning_rate * hidden_outputs{end}' * output_error;
            obj.output_weights = obj.output_weights + obj.prev_output_weight_delta;
        
            
            for i = obj.n_hidden_layers:-1:1
                if i == 1
                    hidden_input = inputs;
                else
                    hidden_input = hidden_outputs{i-1};
                end
                obj.prev_hidden_weight_delta{i} = obj.momentum * obj.prev_hidden_weight_delta{i} - obj.learning_rate * hidden_input' * hidden_errors{i};
                obj.hidden_weights{i} = obj.hidden_weights{i} + obj.prev_hidden_weight_delta{i};
            end
        end

        function [output, hidden_outputs] = forward(obj, input)
           
            hidden_outputs = cell(1, obj.n_hidden_layers);
            for i = 1:obj.n_hidden_layers
                if i == 1
                    hidden_input = input * obj.hidden_weights{i};
                else
                    hidden_input = hidden_outputs{i-1} * obj.hidden_weights{i};
                end
                hidden_outputs{i} = sigmoid(hidden_input);
            end
            output = sigmoid(hidden_outputs{end} * obj.output_weights);
        end
    
        function output = predict(obj, inputs)
            
            [output,~] = obj.forward(inputs);
        end
    
        function set_learning_rate(obj, learning_rate)
            
            obj.learning_rate = learning_rate;
        end
    
        function set_momentum(obj, momentum)
            
            obj.momentum = momentum;
        end
    end
end

function y = sigmoid(x)
    
    y = 1./(1+exp(-x));
end

function y = sigmoid_derivative(x)
    
    y = sigmoid(x) .* (1-sigmoid(x));
end
