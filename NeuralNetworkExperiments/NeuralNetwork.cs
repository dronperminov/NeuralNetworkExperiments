using System;

namespace NeuralNetworkExperiments {
    class NeuralNetwork {
        int inputs;
        int outputs;
        int[] hiddenNeuronsSize; // массив количества нейронов в скрытых слоях

        NeuralLayer inputLayer;
        public readonly NeuralLayer[] hiddenLayers;
        public readonly NeuralLayer outputLayer;

        public NeuralNetwork(int inputs, int outputs, int[] hiddenNeuronsSize, ActivationFunctionType hiddenActivation = ActivationFunctionType.sigmoid, 
            ActivationFunctionType outputActivation = ActivationFunctionType.noChange) {
            this.inputs = inputs;
            this.outputs = outputs;
            this.hiddenNeuronsSize = hiddenNeuronsSize;

            inputLayer = new NeuralLayer(inputs, inputs, NeuralLayerType.input, ActivationFunctionType.noChange);

            hiddenLayers = new NeuralLayer[hiddenNeuronsSize.Length];
            hiddenLayers[0] = new NeuralLayer(inputs, hiddenNeuronsSize[0], NeuralLayerType.hidden, hiddenActivation);

            for (int i = 1; i < hiddenNeuronsSize.Length; i++)
                hiddenLayers[i] = new NeuralLayer(hiddenNeuronsSize[i - 1], hiddenNeuronsSize[i], NeuralLayerType.hidden, hiddenActivation);

            outputLayer = new NeuralLayer(hiddenNeuronsSize[hiddenNeuronsSize.Length - 1], outputs, NeuralLayerType.output, outputActivation);
        }

        public double[] GetOutputs(double[] signals) {
            inputLayer.SetInputs(signals);
            hiddenLayers[0].SetInputs(inputLayer);

            for (int i = 1; i < hiddenNeuronsSize.Length; i++)
                hiddenLayers[i].SetInputs(hiddenLayers[i - 1]);

            outputLayer.SetInputs(hiddenLayers[hiddenNeuronsSize.Length - 1]);

            return outputLayer.GetOutputs();
        }

        public void Train(double[][] learnInputData, double[][] learnOutputData, double alpha, double eps, int maxEpoch, bool log = false, int logInterval = 1000) {
            double error;
            long epoch = 0;

            do {
                error = 0;

                for (int p = 0; p < learnInputData.Length; p++) {
                    double[] results = GetOutputs(learnInputData[p]);
                    double[] sigmas = new double[outputs];

                    for (int i = 0; i < outputs; i++) {
                        sigmas[i] = learnOutputData[p][i] - results[i];
                        error += sigmas[i] * sigmas[i];
                    }

                    double[][] deltas = new double[hiddenNeuronsSize.Length][];

                    for (int i = hiddenNeuronsSize.Length - 1; i >= 0; i--) {
                        if (i == hiddenNeuronsSize.Length - 1) {
                            deltas[i] = hiddenLayers[i].GetErrors(outputLayer, sigmas);
                        }
                        else {
                            deltas[i] = hiddenLayers[i].GetErrors(hiddenLayers[i + 1], deltas[i + 1]);
                        }
                    }

                    for (int layer = 0; layer < hiddenNeuronsSize.Length; layer++) {
                        for (int i = 0; i < hiddenNeuronsSize[layer]; i++) {
                            for (int j = 0; j < hiddenLayers[layer].inputsSize; j++) {
                                double weight = hiddenLayers[layer].GetWeight(j, i);
                                double output = layer == 0 ? inputLayer.GetOutput(j) : hiddenLayers[layer - 1].GetOutput(j);
                                double gradient = hiddenLayers[layer].GetDerivativeOutput(i);

                                hiddenLayers[layer].SetWeight(j, i, weight + alpha * deltas[layer][i] * output * gradient);
                            }
                        }
                    }

                    for (int i = 0; i < outputs; i++) {
                        for (int j = 0; j < hiddenNeuronsSize[hiddenNeuronsSize.Length - 1]; j++) {
                            double weight = outputLayer.GetWeight(j, i);
                            double output = hiddenLayers[hiddenNeuronsSize.Length - 1].GetOutput(j);
                            outputLayer.SetWeight(j, i, weight + alpha * sigmas[i] * output);
                        }
                    }
                }

                if (log && epoch % logInterval == 0) {
                    Log(Math.Sqrt(error), epoch);                    
                }

                epoch++;
            } while (Math.Sqrt(error) > eps && epoch < maxEpoch);
        }

        public void Log(double error, long epoch) {
            inputLayer.PrintState();

            for (int i = 0; i < hiddenNeuronsSize.Length; i++)
                hiddenLayers[i].PrintState();

            outputLayer.PrintState();

            Console.WriteLine("Error: {0}", error);
            Console.WriteLine("Epoch: {0}", epoch);
        }
    }
}