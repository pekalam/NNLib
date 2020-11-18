using System;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using NNLib.ActivationFunction;
using NNLib.Data;

namespace NNLib.MLP
{
    public static class MLPNetworkExtensions
    {
        public static double CalculateOutput(this MLPNetwork network,double x)
        {
            if (network.Layers[0].InputsCount != 1 || network.Layers[^1].NeuronsCount != 1)
            {
                throw new Exception("Network must have 1 neuron at output and 1 input at input layer");
            }

            network.CalculateOutput(Matrix<double>.Build.Dense(1, 1, x));

            return network.Output!.At(0, 0);
        }
    }

    public class MLPNetwork : Network<PerceptronLayer>
    {
        public MLPNetwork(params PerceptronLayer[] perceptronLayers) : base(perceptronLayers)
        {
        }

        public static MLPNetwork Create(int inputs,
            params (int neuronsCount, IActivationFunction activationFunction, MatrixBuilder builder)[] layers)
        {
            var netLayers = new PerceptronLayer[layers.Length];
            var inputLayer = new PerceptronLayer(inputs, layers[0].neuronsCount, layers[0].activationFunction, layers[0].builder);

            netLayers[0] = inputLayer;

            for (int i = 1; i < layers.Length; i++)
            {
                var layer = new PerceptronLayer(netLayers[i - 1].NeuronsCount, layers[i].neuronsCount,
                    layers[i].activationFunction, layers[i].builder);
                netLayers[i] = layer;
            }

            var net = new MLPNetwork(netLayers);
            return net;
        }

        public static MLPNetwork Create(int inputs,
            params (int neuronsCount, IActivationFunction activationFunction)[] layers)
        {
            var x = layers.Select(l => (l.neuronsCount, l.activationFunction, (MatrixBuilder)new DefaultNormDistMatrixBuilder()))
                .ToArray();
            return Create(inputs, x);
        }

        public MLPNetwork Clone() => new MLPNetwork(Layers.Select(l => l.Clone()).ToArray());

        internal override PerceptronLayer CreateHiddenLayer(int inputsCount, int neuronsCount) => new PerceptronLayer(inputsCount, neuronsCount, new SigmoidActivationFunction());

        internal override PerceptronLayer CreateOutputLayer(int inputsCount, int neuronsCount) => new PerceptronLayer(inputsCount, neuronsCount, new LinearActivationFunction());

        public override void CalculateOutput(Matrix<double> input)
        {
            Matrix<double> prevLayerOutput = input;
            for (int l = 0; l < _layers.Count; ++l)
            {
                _layers[l].CalculateOutput(prevLayerOutput!);
                prevLayerOutput = _layers[l].Output!;
            }

            Output = prevLayerOutput;
        }
    }
}