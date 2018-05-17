using System;

namespace NeuralNetworkExperiments {
    enum NeuralLayerType {
        input,
        hidden,
        output
    }

    class NeuralLayer {
        NeuralLayerType type;
        public readonly int outputsSize;
        public readonly int inputsSize;

        public readonly Neuron[] neurons;

        public NeuralLayer(int inputsSize, int outputsSize, NeuralLayerType type, ActivationFunctionType activationType = ActivationFunctionType.sigmoid) {
            this.inputsSize = inputsSize;
            this.outputsSize = outputsSize;
            this.type = type;

            neurons = new Neuron[outputsSize];

            for (int i = 0; i < outputsSize; i++) {
                switch (type) {
                    case NeuralLayerType.input:
                        neurons[i] = new Neuron(NeuronType.input);
                        break;

                    case NeuralLayerType.hidden:
                        neurons[i] = new Neuron(NeuronType.hidden, inputsSize, activationType);
                        break;

                    case NeuralLayerType.output:
                        neurons[i] = new Neuron(NeuronType.output, inputsSize, activationType);
                        break;
                }
            }
        }

        // установка входов слоя
        public void SetInputs(double[] signals) {
            if (type == NeuralLayerType.input) {
                for (int i = 0; i < outputsSize; i++)
                    neurons[i].SetInput(0, signals[i]);
            }
            else {
                for (int i = 0; i < outputsSize; i++)
                    for (int j = 0; j < inputsSize; j++) {
                        neurons[i].SetInput(j, signals[j]);
                    }
            }
        }

        // передача сигналов от слоя layer к этому
        public void SetInputs(NeuralLayer layer) {
            double[] outputs = layer.GetOutputs();

            for (int i = 0; i < outputsSize; i++) {
                for (int j = 0; j < inputsSize; j++)
                    neurons[i].SetInput(j, outputs[j]);
            }
        }
        
        public double GetWeight(int i, int j) {
            return neurons[j].GetWeight(i);
        }

        public void SetWeight(int i, int j, double weight) {
            neurons[j].SetWeight(i, weight);
        }

        public double[] GetErrors(NeuralLayer nextLayer, double[] nextErrors) {
            double[] errors = new double[outputsSize];

            for (int i = 0; i < outputsSize; i++) {
                errors[i] = 0;

                for (int j = 0; j < nextErrors.Length; j++)
                    errors[i] += nextErrors[j] * nextLayer.GetWeight(i, j);
            }

            return errors;
        }

        public double GetOutput(int index) {
            return neurons[index].GetOutput();
        }

        public double GetDerivativeOutput(int index) {
            return neurons[index].GetDerivativeOutput();
        }

        public double[] GetOutputs() {
            double[] outputs = new double[outputsSize];

            for (int i = 0; i < outputsSize; i++)
                outputs[i] = neurons[i].GetOutput();

            return outputs;
        }

        public void PrintState() {
            if (type == NeuralLayerType.input) {
                Console.Write("Input");
            }
            else if (type == NeuralLayerType.hidden) {
                Console.Write("Hidden");
            }
            else if (type == NeuralLayerType.output) {
                Console.Write("Output");
            }

            Console.WriteLine(" neural layer");
            Console.WriteLine("Inputs count: {0}", inputsSize);
            Console.WriteLine("Output counts: {0}", outputsSize);
            Console.WriteLine("Neurons:");

            for (int i = 0; i < neurons.Length; i++)
                neurons[i].Print();

            Console.WriteLine();
        }
    }
}
