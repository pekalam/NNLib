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
                throw new Exception("Network must have 1 neuron at output and 1 on input layers");
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

        public PerceptronLayer InsertAfter(int ind)
        {
            ind++;
            if (ind > TotalLayers || ind < 0) throw new ArgumentException("Cannot insert after " + ind + " - index out of bounds");

            var layer = new PerceptronLayer(ind == 0 ? Layers[0].InputsCount : Layers[ind-1].NeuronsCount, ind == TotalLayers ? Layers[^1].NeuronsCount : Layers[ind].InputsCount, new LinearActivationFunction());

            
            _layers.Insert(ind, layer);
            layer.AssignNetwork(this);
            AssignEventHandlers(layer);
            if (!layer.IsInitialized)
            {
                layer.Initialize();
            }
            layer.InitializeMemory();
            RaiseNetworkStructureChanged();
            return layer;
        }

        public PerceptronLayer InsertBefore(int ind) => InsertAfter(ind - 1);

        public MLPNetwork Clone() => new MLPNetwork(Layers.Select(l => l.Clone()).ToArray());

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