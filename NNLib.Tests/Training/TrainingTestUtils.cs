using NNLib.ActivationFunction;
using NNLib.Common;
using NNLib.Data;
using NNLib.MLP;
using M = MathNet.Numerics.LinearAlgebra.Matrix<double>;

namespace NNLib.Tests
{
    public static class TrainingTestUtils
    {
        public static MLPNetwork CreateNetwork(int inputs,
            params (int neuronsCount, IActivationFunction activationFunction)[] layers)
        {
            var netLayers = new PerceptronLayer[layers.Length];
            var inputLayer = new PerceptronLayer(inputs, layers[0].neuronsCount, layers[0].activationFunction);

            netLayers[0] = inputLayer;

            for (int i = 1; i < layers.Length; i++)
            {
                var layer = new PerceptronLayer(netLayers[i - 1].NeuronsCount, layers[i].neuronsCount,
                    layers[i].activationFunction);
                netLayers[i] = layer;
            }

            var net = new MLPNetwork(netLayers);
            return net;
        }

        public static Data.SupervisedTrainingSamples AndGateSet()
        {
            var input = new[]
            {
                new []{0d,0d},
                new []{0d,1d},
                new []{1d,0d},
                new []{1d,1d},
            };

            var expected = new[]
            {
                new []{0d},
                new []{0d},
                new []{0d},
                new []{1d},
            };

            return Data.SupervisedTrainingSamples.FromArrays(input, expected);
        }

        public static bool CompareTo(this M m1, M m2)
        {
            if (m1.RowCount != m2.RowCount || m1.ColumnCount != m2.ColumnCount)
            {
                return false;
            }

            for (int i = 0; i < m1.RowCount; i++)
            {
                for (int j = 0; j < m1.ColumnCount; j++)
                {
                    if (m1[i, j] != m2[i, j])
                    {
                        return false;
                    }
                }
            }

            return true;
        }
    }
}