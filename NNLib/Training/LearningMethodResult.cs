using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace NNLib
{
    public class LearningMethodResult
    {
        private LearningMethodResult()
        {
        }

        public Matrix<double>[] Weigths { get; set; }
        public Matrix<double>[] Biases { get; set; }

        public static LearningMethodResult FromNetwork(MLPNetwork network)
        {
            return new LearningMethodResult()
            {
                Weigths = new Matrix<double>[network.TotalLayers],
                Biases = new Matrix<double>[network.TotalLayers],
            };
        }

        public LearningMethodResult Empty(MLPNetwork net)
        {
            for (int i = 0; i < Weigths.Length; i++)
            {
                Weigths[i] = Matrix<double>.Build.Dense(net.Layers[i].NeuronsCount, net.Layers[i].InputsCount, 0d);
                Biases[i] = Matrix<double>.Build.Dense(net.Layers[i].NeuronsCount, 1, 0d);
            }
            return this;
        }
    }
}