using MathNet.Numerics.LinearAlgebra;
using System.Linq;

namespace NNLib
{
    public class MLPNetwork : Network<PerceptronLayer>
    {
        public MLPNetwork(params PerceptronLayer[] perceptronLayers) : base(perceptronLayers)
        {
        }

        public MLPNetwork Clone() => new MLPNetwork(Layers.Select(l => l.Clone()).ToArray());

        public override void CalculateOutput(Matrix<double> input)
        {
            Matrix<double> prevLayerOutput = input;
            for (int l = 0; l < Layers.Count; ++l)
            {
                Layers[l].CalculateOutput(prevLayerOutput);
                prevLayerOutput = Layers[l].Output;
            }

            Output = prevLayerOutput;
        }
    }
}