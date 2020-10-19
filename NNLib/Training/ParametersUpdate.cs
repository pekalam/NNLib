using MathNet.Numerics.LinearAlgebra;
using NNLib.MLP;

#pragma warning disable 8618

namespace NNLib.Training
{
    /// <summary>
    /// Stores update calculated by algorithm for network parameters
    /// </summary>
    public class ParametersUpdate
    {
        private ParametersUpdate()
        {
        }

        public Matrix<double>[] Weights;
        public Matrix<double>[] Biases;

        public static ParametersUpdate FromNetwork(MLPNetwork network)
        {
            var update = new ParametersUpdate()
            {
                Weights = new Matrix<double>[network.TotalLayers],
                Biases = new Matrix<double>[network.TotalLayers],
            };

            for (int i = 0; i < update.Weights.Length; i++)
            {
                update.Weights[i] = Matrix<double>.Build.Dense(network.Layers[i].Weights.RowCount,
                    network.Layers[i].Weights.ColumnCount);
                update.Biases[i] = Matrix<double>.Build.Dense(network.Layers[i].Biases.RowCount,
                    network.Layers[i].Biases.ColumnCount);
            }

            return update;
        }
    }
}