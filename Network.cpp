#include<vector>
#include<iostream>
#include<cstdlib>
#include<cassert>
#include<cmath>

using namespace std;


// ##########################
// ####### Data Class #######
// ##########################

// Class for handling data generation and iteration
class Data {
public:
	Data(unsigned num_samples);
	void generate_data(unsigned num_samples);
	vector<double> next_input();
	vector<double> next_target();
	vector<vector<double> > inputs;
	vector<vector<double> > targets;
	static bool end;
private:
	unsigned num_samples;
	static unsigned iter;
	
};

unsigned Data::iter = 0;
bool Data::end = 0;

// Class constructor
Data::Data(unsigned num_samples) {
	// Generates data for the XOR problem
	num_samples = num_samples;
	for (int sample = num_samples; sample > 0; --sample) {
		vector<double> s;
		int n1 = int(2.0 * rand() / double(RAND_MAX));
		int n2 = int(2.0 * rand() / double(RAND_MAX));
		int t = n1 ^ n2;
		s.push_back(n1);
		s.push_back(n2);
		inputs.push_back(s);
		vector<double> tmp;
		tmp.push_back(t);
		targets.push_back(tmp);
	}
}

// Returns the next data input to be used in the forward computation
vector<double> Data::next_input(){
	if (iter == (num_samples - 1)){
		end = 1;
	}
	vector<double> out = inputs.back();
	// Datapoint is remove from input container
	inputs.pop_back();

	return out;
}

// Returns the next data target for backpropagation
vector<double> Data::next_target(){
	if (iter == (num_samples - 1)){
		end = 1;
	}
	vector<double> out = targets.back();
	// Datapoint is removed from target container
	targets.pop_back();

	return out;
}

// A simple container for storing the connection weights between neurons in each layer
// This allows for a more intuitive code base, with easier-to-read syntax
struct Connection
{
	double weight;
};


// ##########################
// ###### Neuron Class ######
// ##########################

// Define an empty neuron class to facilitate the forward definition of a Layer container
// this is simply a vector of neurons which is intuitively referred to as a Layer in code
// The forward definition is required as the Layer container is refrenced in the Neuron code
class Neuron;
typedef vector<Neuron> Layer;	

// Next, the actual Neuron class is redefined
class Neuron {

public:
	Neuron(unsigned n_output, unsigned neur_idx);
	void forward(const Layer &input_layer);
	void set_output(double value);
	double get_output() const;
	void out_grad_comp(double target);
	void hidd_grad_comp(const Layer &next);
	void update(Layer &previous);
	double output;
private:
	static double weight_init() { return rand() / double(RAND_MAX); };
	static double activation(double inp);
	static double activation_derivative(double inp);
	static double lr;
	vector<Connection> weights;
	unsigned neur_idx;
	double gradient;
};

// Learning rate of each neuron within the network
double Neuron::lr = 0.2;

// Neuron Constructor
Neuron::Neuron(unsigned n_output, unsigned neur_idx) {
	// Iterate over number of neurons within layer, appending a connection onto the weight container of the neuron
	for (unsigned conn = 0; conn < n_output; ++conn) {
		weights.push_back(Connection());
		// Initialize the weights using a random number
		weights.back().weight = weight_init();
	}
}

// Neuron function to set the value of the neuron
void Neuron::set_output(double value) {
	output = value;
}

// Neuron function to return the output value of the neuron
double Neuron::get_output() const {
	return output;
}

// A non-linear activation function to be applied at each neuron
double Neuron::activation(double inp) {
	// Hyperbolic tangent activation
	return tanh(inp);
}

// A derivative of the activation function, used in backpropagation
double Neuron::activation_derivative(double inp) {
	// Activation derivative (Approximation)
	return 1.0 - inp * inp;
}

// Function for computing the gradient of the output layer neurons, used in backpropagation
void Neuron::out_grad_comp(double target) {
	gradient = (target - output) * Neuron::activation_derivative(output);
}

// Function for computing the gradient of the hidden layer neurons, used in backpropagation
void Neuron::hidd_grad_comp(const Layer &next) {
	// Compute gradient w.r.t to the sum of outputs
	double out_sum = 0.0;
	// Iterate over the outgoing connection from the current neuron, summing the product of these weights with the next layer's gradient
	for (unsigned neur_n = 0; neur_n < next.size() - 1; ++neur_n) {
		out_sum += weights[neur_n].weight * next[neur_n].gradient;
	}

	gradient = out_sum * Neuron::activation_derivative(output);
}


// Gradient descent update step, iteratively updates each neuron within the given layer
void Neuron::update(Layer &previous) {
	for (unsigned neur_n = 0; neur_n < previous.size(); ++neur_n) {
		// Update each neuron's weight
		previous[neur_n].weights[neur_idx].weight += (lr * previous[neur_n].get_output() * gradient);
	}
}

// A forward pass over the neuron,
// sums previous layer inputs into neuron and passes it through a non-linear activation function
void Neuron::forward(const Layer &input_layer) {
	double input_sum = 0.0;
	// Loop over neurons from previous layer to obtain the sum of each output multiplied with the neuron weights
	for (unsigned prev_neuron = 0; prev_neuron < input_layer.size(); ++prev_neuron) {
		input_sum += input_layer[prev_neuron].get_output() * input_layer[prev_neuron].weights[neur_idx].weight;
	}

	// Apply a non-linear activation function to the sum of neuron inputs
	output = Neuron::activation(input_sum);
}



// ###########################
// ###### Network Class ######
// ###########################



class Network {

public:
	Network(const vector<unsigned> &topology);
	void forward(const vector<double> &input);
	void backward(const vector<double> &targets);
	void results(vector<double> &results) const;
private:
	vector<Layer> n_layers;
	double error_sum;
};


// Class constructor
Network::Network(const vector<unsigned> &topology) {
	// Obtain the number of layers in network from the length of topology vector
	unsigned num_layers = topology.size();
	// Iterate over the number of layers, adding each one to the n_layers container which stores the layers
	for (unsigned layer = 0; layer < num_layers; ++layer) {
		n_layers.push_back(Layer());
		// Obtain the number of outputs of each layer up until the last layer which has 0
		unsigned n_outputs = layer != topology.size() ? topology[layer] : 0;
		// For each layer, now iterate over the number of neurons (+ 1 for the bias) contained within this layer, adding these to the Layer container above
		std::cout << "Layer created!" << std::endl;
		for (unsigned neuron = 0; neuron < topology[layer] + 1; ++neuron) {
			// Append neurons onto the newest (last) element in the n_layers container which is being filled by the outer loop
			n_layers.back().push_back(Neuron(n_outputs, neuron));
			std::cout << "Neuron Created!" << std::endl;
		}
		// Assign a constant 1.0 output for each bias neuron
		n_layers.back().back().set_output(1.0);
	}
}

// Updates the results buffer with the newest network result
void Network::results(vector<double> &results) const {

	results.clear();

	for (unsigned neur_n = 0; neur_n < n_layers.back().size() - 1; ++neur_n) {
		results.push_back(n_layers.back()[neur_n].output);
	}
}

// Forward propagation steps of the network, producing an output for a given input
void Network::forward(const vector<double> &input) {
	// First, ensure that the number of input elements matches the number of neurons in input layer
	assert(input.size() == n_layers[0].size() - 1);
	// Each input element is passed into each neuron in the INPUT LAYER
	for (unsigned inp_i = 0; inp_i < input.size(); ++inp_i) {
		n_layers[0][inp_i].set_output(input[inp_i]);
	}
	// Forward propagation through the HIDDEN LAYERS
	for (unsigned layer = 1; layer < n_layers.size(); ++layer) {
		Layer &input_layer = n_layers[layer - 1];
		for (unsigned neuron = 0; neuron < n_layers[layer].size() - 1; ++neuron) {
			// Call forward function on each neuron and its connections
			n_layers[layer][neuron].forward(input_layer);
		}
	}
}

// Backpropagation steps for the network, moving backward from the target values to obtain neuron derivatives for training
void Network::backward(const vector<double> &targets) {
	error_sum = 0.0;
	// Compute loss of each neuron in output layer (RMS)
	for (unsigned n_out = 0; n_out < n_layers.back().size() - 1; ++n_out) {
		double diff = targets[n_out] - n_layers.back()[n_out].get_output();
		error_sum += diff * diff;
	}
	error_sum = sqrt(error_sum / (n_layers.back().size() - 1));
	

	// Gradient Computation using the "grad_comp" method from the Neuron class
	// Output layer (Iterate over each neuron and compute gradient)
	for (unsigned n_out = 0; n_out < n_layers.back().size() - 1; ++n_out) {
		n_layers.back()[n_out].out_grad_comp(targets[n_out]);
	}
	// Hidden layers (Iterate over each layer moving backwards, compute gradient at each neuron)
	for (unsigned layer_n = n_layers.size() - 2; layer_n > 0; --layer_n) {
		for (unsigned neuron_n = 0; neuron_n < n_layers[layer_n].size(); ++neuron_n) {
			n_layers[layer_n][neuron_n].hidd_grad_comp(n_layers[layer_n + 1]);
		}
	}
	// Finally, perform a weight update for all weights within the network using the computed gradient above
	for (unsigned layer_n = n_layers.size() - 1; layer_n > 0; --layer_n) {
		for (unsigned neur_n = 0; neur_n < n_layers[layer_n].size(); ++neur_n)
			n_layers[layer_n][neur_n].update(n_layers[layer_n - 1]);
	}
}

int main() {


	// Generate 1000 training data samples
	unsigned samples = 1000;
	Data Data(samples);
	
	// Generate a three layer network with 2, 4, 1 neurons in each layer respectively
	vector<unsigned> topology;
	topology.push_back(2);
	topology.push_back(4);
	topology.push_back(1);

	// Initialize the network using the topology vector
	Network Test_Network(topology);

	// Iterate over data container until they are empty, stated by the .end boolean
	while(Data.end != 1){
		vector<double> inp;
		inp = Data.next_input();
		cout << "Input: " << inp[0] << "," << inp[1] << endl;
		// FORWARD PASS
		Test_Network.forward(inp);

		vector<double> results;
		Test_Network.results(results);
		cout << results[0] << endl;

		vector<double> target = Data.next_target();
		cout << "Target: " << target[0] << endl;

		// BACKWARD PASS
		Test_Network.backward(target);

	}

	cout << "DONE" << endl;

}