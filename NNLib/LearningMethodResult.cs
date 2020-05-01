using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace NNLib
{
    public class LearningMethodResult
    {
        private LearningMethodResult()
        {
        }

        public List<Matrix<double>> Weigths { get; set; }
        public List<Matrix<double>> Biases { get; set; }

        public static LearningMethodResult FromNetwork(MLPNetwork network)
        {
            return new LearningMethodResult()
            {
                Weigths = new List<Matrix<double>>(Enumerable.Repeat<Matrix<double>>(null, network.TotalLayers)),
                Biases = new List<Matrix<double>>(Enumerable.Repeat<Matrix<double>>(null, network.TotalLayers)),
            };
        }
    }
}