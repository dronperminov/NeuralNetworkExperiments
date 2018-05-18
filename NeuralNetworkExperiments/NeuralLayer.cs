using System;

namespace NeuralNetworkExperiments {
    enum NeuralLayerType {
        input, // входной слой
        hidden, // скрытй слой
        output // выходной слой
    }

    enum ActivationFunctionType {
        sigmoid, // логистическая функция
        tangent, // гиперболический тангенс
        relu, // выпрямитель
        noChange // без изменений
    }

    class NeuralLayer {
        NeuralLayerType type; // тип слоя
        ActivationFunctionType activationType; // тип функции активации

        public readonly int inputsSize; // число входов 
        public readonly int outputsSize; // число выходов слоя

        double[] inputs; // входы (значения сигналов на входе
        double[][] weights; // матрица весов

        public NeuralLayer(int inputsSize, int outputsSize, NeuralLayerType type, ActivationFunctionType activationType = ActivationFunctionType.sigmoid) {
            this.inputsSize = inputsSize;
            this.outputsSize = outputsSize;

            this.type = type;

            this.activationType = activationType;

            Random random = new Random();

            inputs = new double[inputsSize];
            weights = new double[outputsSize][];

            for (int i = 0; i < outputsSize; i++) {
                weights[i] = new double[inputsSize];

                for (int j = 0; j < inputsSize; j++) {
                    if (type == NeuralLayerType.input) {
                        weights[i][j] = i == j ? 1 : 0;
                    }
                    else {
                        weights[i][j] = random.NextDouble();
                    }
                }
            }
        }

        // установка входов слоя
        public void SetInputs(double[] signals) {
            for (int i = 0; i < inputsSize; i++)
                inputs[i] = signals[i];
        }

        // передача сигналов от слоя layer к этому
        public void SetInputs(NeuralLayer layer) {
            double[] outputs = layer.GetOutputs();

            for (int i = 0; i < inputsSize; i++)
                inputs[i] = outputs[i];
        }
        
        // получение веса
        public double GetWeight(int i, int j) {
            return weights[j][i];
        }

        // изменение веса
        public void SetWeight(int i, int j, double weight) {
            weights[j][i] = weight;
        }

        // получение массива ошибок относительно следующего слоя
        public double[] GetErrors(NeuralLayer nextLayer, double[] nextErrors) {
            double[] errors = new double[outputsSize];

            for (int i = 0; i < outputsSize; i++) {
                errors[i] = 0;

                for (int j = 0; j < nextErrors.Length; j++)
                    errors[i] += nextErrors[j] * nextLayer.GetWeight(i, j);
            }

            return errors;
        }

        // получение выхода слоя по индексу index
        public double GetOutput(int index) {
            double sum = 0;

            for (int i = 0; i < inputsSize; i++)
                sum += inputs[i] * weights[index][i];

            return ActivationFunction(sum);
        }

        public double GetDerivativeOutput(int index) {
            double sum = 0;

            for (int i = 0; i < inputsSize; i++)
                sum += inputs[i] * weights[index][i];

            return ActivationFunctionDerivative(sum);
        }

        public double[] GetOutputs() {
            double[] outputs = new double[outputsSize];

            for (int i = 0; i < outputsSize; i++) {
                outputs[i] = 0;

                for (int j = 0; j < inputsSize; j++)
                    outputs[i] += inputs[j] * weights[i][j];

                if (type != NeuralLayerType.input)
                    outputs[i] = ActivationFunction(outputs[i]);
            }

            return outputs;
        }

        double Sigmoid(double x) { return 1.0 / (1 + Math.Exp(-x)); }
        double SigmoidDerivative(double x) { return Sigmoid(x) * (1 - Sigmoid(x)); }

        double HiperbolicTangent(double x) { return (Math.Exp(2 * x) - 1) / (Math.Exp(2 * x) + 1); }
        double HiperbolicTangentDerivative(double x) { return 4 * Math.Exp(2 * x) / Math.Pow(Math.Exp(2 * x) + 1, 2); }

        double ReLU(double x) { return x > 0 ? x : 0; }
        double ReLUDerivative(double x) { return x > 0 ? 1 : 0; }

        double NoChange(double x) { return x; }
        double NoChangeDerivative(double x) { return 1; }

        double ActivationFunction(double x) {
            switch (activationType) {
                case ActivationFunctionType.sigmoid:
                    return Sigmoid(x);

                case ActivationFunctionType.tangent:
                    return HiperbolicTangent(x);

                case ActivationFunctionType.relu:
                    return ReLU(x);

                case ActivationFunctionType.noChange:
                    return NoChange(x);
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

                case ActivationFunctionType.noChange:
                    return NoChangeDerivative(x);
            }

            return 1;
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
            Console.WriteLine("    Inputs count: {0}", inputsSize);
            Console.WriteLine("    Output counts: {0}", outputsSize);

            Console.Write("    Inputs: [ ");

            for (int i = 0; i < inputsSize; i++)
                Console.Write("{0} ", inputs[i]);

            Console.WriteLine("]");

            Console.Write("    Outputs: [ ");

            for (int i = 0; i < outputsSize; i++)
                Console.Write("{0} ", GetOutput(i));

            Console.WriteLine("]");

            Console.WriteLine("    Weights: ");

            for (int i = 0; i < outputsSize; i++) {
                Console.Write("    ");
                for (int j = 0; j < inputsSize; j++)
                    Console.Write(weights[i][j].ToString("0.####") + "  ");

                Console.WriteLine();
            }

            Console.WriteLine();
        }
    }
}
