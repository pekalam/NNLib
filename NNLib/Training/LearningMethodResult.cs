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
    }
}