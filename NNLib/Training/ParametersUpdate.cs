using System.Linq;
using MathNet.Numerics.LinearAlgebra;
#pragma warning disable 8618

namespace NNLib
{
    public class ParametersUpdate
    {
        private ParametersUpdate()
        {
        }

        public Matrix<double>[] Weights;
        public Matrix<double>[] Biases;

        public static ParametersUpdate FromNetwork(MLPNetwork network)
        {
            return new ParametersUpdate()
            {
                Weights = new Matrix<double>[network.TotalLayers],
                Biases = new Matrix<double>[network.TotalLayers],
            };
        }
    }
}