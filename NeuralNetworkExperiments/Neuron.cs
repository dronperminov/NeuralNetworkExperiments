using System;

namespace NeuralNetworkExperiments {
    enum NeuronType {
        input,
        hidden,
        output
    }

    enum ActivationFunctionType {
        sigmoid,
        tangent,
        relu
    }

    class Neuron {
        int inputsSize; // число входов нейрона
        NeuronType type; // тип нейрона
        ActivationFunctionType activationType; // тип активационной функции
        double[] inputs; // входные сигналы
        double[] weights; // весы на связи
        Random random;

        public Neuron(NeuronType type, int inputsSize = 1) {
            this.inputsSize = inputsSize;
            this.type = type;
            activationType = ActivationFunctionType.sigmoid;

            inputs = new double[inputsSize];
            weights = new double[inputsSize];

            if (type != NeuronType.input) {
                random = new Random();

                for (int i = 0; i < inputsSize; i++)
                    weights[i] = -0.5 + random.NextDouble();
            }
            else {
                for (int i = 0; i < inputsSize; i++)
                    weights[i] = 1;
            }
        }

        public Neuron(NeuronType type, int inputsSize, ActivationFunctionType activationType) {
            this.inputsSize = inputsSize;
            this.type = type;
            this.activationType = activationType;

            inputs = new double[inputsSize];
            weights = new double[inputsSize];

            if (type != NeuronType.input) {
                random = new Random();

                for (int i = 0; i < inputsSize; i++)
                    weights[i] = -0.5 + random.NextDouble();
            }
            else {
                for (int i = 0; i < inputsSize; i++)
                    weights[i] = 1;
            }
        }

        public double[] GetInputs() { return inputs; }
        public void SetInputs(double[] inputs) { this.inputs = inputs; }

        public double[] GetWeights() { return weights; }
        public void SetWeights(double[] weights) { this.weights = weights; }

        public double GetInput(int index) { return inputs[index]; }
        public void SetInput(int index, double input) { inputs[index] = input; }

        public double GetWeight(int index) { return weights[index]; }
        public void SetWeight(int index, double weight) { weights[index] = weight; }

        double Sigmoid(double x) { return 1.0 / (1 + Math.Exp(-x)); }
        double SigmoidDerivative(double x) { return Sigmoid(x) * (1 - Sigmoid(x)); }

        double HiperbolicTangent(double x) { return (Math.Exp(2 * x) - 1) / (Math.Exp(2 * x) + 1); }
        double HiperbolicTangentDerivative(double x) { return 4 * Math.Exp(2 * x) / Math.Pow(Math.Exp(2 * x) + 1, 2); }

        double ReLU(double x) { return x > 0 ? x : 0; }
        double ReLUDerivative(double x) { return x > 0 ? 1 : 0; }

        double ActivationFunction(double x) {
            switch (activationType) {
                case ActivationFunctionType.sigmoid:
                    return Sigmoid(x);

                case ActivationFunctionType.tangent:
                    return HiperbolicTangent(x);

                case ActivationFunctionType.relu:
                    return ReLU(x);
            }

            return x;
        }

        double ActivationFunctionDerivative(double x) {
            switch (activationType) {
                case ActivationFunctionType.sigmoid:
                    return SigmoidDerivative(x);

                case ActivationFunctionType.tangent:
                    return HiperbolicTangentDerivative(x);

                case ActivationFunctionType.relu:
                    return ReLUDerivative(x);
            }

            return 1;
        }

        public double GetOutput() {
            double sum = 0;

            for (int i = 0; i < inputsSize; i++)
                sum += inputs[i] * weights[i];

            return type == NeuronType.hidden ? ActivationFunction(sum) : sum;
        }

        public double GetDerivativeOutput() {
            double sum = 0;

            for (int i = 0; i < inputsSize; i++)
                sum += inputs[i] * weights[i];

            return ActivationFunctionDerivative(sum);
        }

        public void Print() {
            if (type == NeuronType.input) {
                Console.Write("InputNeuron: ");
            }
            else if (type == NeuronType.hidden) {
                Console.Write("HiddenNeuron: ");
            }
            else {
                Console.Write("OutputNeuron: ");
            }

            Console.Write("inputs: [ ");
            for (int i = 0; i < inputsSize; i++)
                Console.Write("{0} ", inputs[i]);

            Console.Write("], weights: [ ");
            for (int i = 0; i < inputsSize; i++)
                Console.Write("{0} ", weights[i]);

            double sum = 0;
            for (int i = 0; i < inputsSize; i++)
                sum += inputs[i] * weights[i];

            Console.WriteLine("], out: {0}", GetOutput());
        }
    }
}
