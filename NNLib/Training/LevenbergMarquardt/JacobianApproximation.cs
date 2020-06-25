using System;
using MathNet.Numerics.LinearAlgebra;

namespace NNLib
{
    public static class JacobianApproximation
    {
        private const double StepSize = 0.001;

        public static Matrix<double> CalcJacobian(MLPNetwork network, ILossFunction lossFunction,
            Matrix<double> input, Matrix<double> expected)
        {
            network.CalculateOutput(input);
            var y = lossFunction.Derivative(network.Output, expected);
            var J = Matrix<double>.Build.Dense(expected.RowCount, network.TotalSynapses + network.TotalBiases);

            int colPos = 0;
            for (int j = 0; j < network.TotalLayers; j++)
            {
                var w = network.Layers[j].Weights;
                for (int k = 0; k < w.RowCount; k++)
                {
                    for (int l = 0; l < w.ColumnCount; l++)
                    {
                        var pw = w[k, l];
                        var del = StepSize * (1 + Math.Abs(pw));
                        w[k, l] += del;
                        network.CalculateOutput(input);
                        var y1 = lossFunction.Derivative(network.Output, expected);
                        var d = (y1 - y).Divide(del);
                        for (int i = 0; i < d.RowCount; i++)
                        {
                            J[i, colPos] = d[i, 0];
                        }

                        w[k, l] = pw;
                        colPos++;
                    }
                }
            }

            for (int j = 0; j < network.TotalLayers; j++)
            {
                var b = network.Layers[j].Biases;
                for (int i = 0; i < b.RowCount; i++)
                {
                    var pb = b[i, 0];
                    var del = StepSize * (1 + Math.Abs(pb));
                    b[i, 0] += del;
                    network.CalculateOutput(input);
                    var y1 = lossFunction.Derivative(network.Output, expected);
                    var d = (y1 - y).Divide(del);
                    for (int k = 0; k < d.RowCount; k++)
                    {
                        J[k, colPos] = d[k, 0];
                    }

                    b[i, 0] = pb;
                    colPos++;
                }
            }


            return J;
        }
    }
}