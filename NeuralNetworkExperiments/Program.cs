using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkExperiments {
    class Program {
        static void Main(string[] args) {
            int inputs = 2;
            int outputs = 1;
            double alpha = 0.1;
            double eps = 1e-7;
            int maxEpochs = 100000;

            int[] neurons = { 4, 2 };

            double[][] trainIn = {
                new double[] { 0, 0 },
                new double[] { 0, 1 },
                new double[] { 1, 0 },
                new double[] { 1, 1 }
            };

            double[][] trainOut = {
                new double[] { 0 },
                new double[] { 1 },
                new double[] { 1 },
                new double[] { 0 }
            };

            NeuralNetwork network = new NeuralNetwork(inputs, outputs, neurons, ActivationFunctionType.sigmoid);
            network.Train(trainIn, trainOut, alpha, eps, maxEpochs, true, 10);

            for (int i = 0; i < trainIn.Length; i++) {
                double[] result = network.GetOutputs(trainIn[i]);

                Console.WriteLine("{0} XOR {1} = {2}", trainIn[i][0], trainIn[i][1], result[0]);
            }

            Console.ReadKey();

        }
    }
}
