using MathNet.Numerics.LinearAlgebra;

namespace NNLib
{
    public class MLPNetwork : Network<PerceptronLayer>
    {

        public MLPNetwork(params PerceptronLayer[] perceptronLayers) : base(perceptronLayers)
        {
        }

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