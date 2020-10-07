using System;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace NNLib
{
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

            return layer;
        }

        public PerceptronLayer InsertBefore(int ind) => InsertAfter(ind - 1);

        public MLPNetwork Clone() => new MLPNetwork(Layers.Select(l => l.Clone()).ToArray());

        public override void CalculateOutput(Matrix<double> input)
        {
            Matrix<double> prevLayerOutput = input;
            for (int l = 0; l < Layers.Count; ++l)
            {
                Layers[l].CalculateOutput(prevLayerOutput!);
                prevLayerOutput = Layers[l].Output!;
            }

            Output = prevLayerOutput;
        }
    }
}