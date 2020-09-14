using System.Linq;
using MathNet.Numerics.LinearAlgebra;
#pragma warning disable 8618

namespace NNLib
{
    public class LearningMethodResult
    {
        private LearningMethodResult()
        {
        }

        public Matrix<double>[] Weights { get; set; }
        public Matrix<double>[] Biases { get; set; }

        public static LearningMethodResult FromNetwork(MLPNetwork network)
        {
            return new LearningMethodResult()
            {
                Weights = new Matrix<double>[network.TotalLayers],
                Biases = new Matrix<double>[network.TotalLayers],
            };
        }

        public LearningMethodResult Empty(MLPNetwork net)
        {
            for (var i = 0; i < Weights.Length; i++)
            {
                Weights[i] = Matrix<double>.Build.Dense(net.Layers[i].NeuronsCount, net.Layers[i].InputsCount, 0d);
                Biases[i] = Matrix<double>.Build.Dense(net.Layers[i].NeuronsCount, 1, 0d);
            }
            return this;
        }
    }
}