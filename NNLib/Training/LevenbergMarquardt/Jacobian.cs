using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Storage;
using NNLib.Data;
using NNLib.LossFunction;
using NNLib.MLP;

namespace NNLib.Training.LevenbergMarquardt
{
    public class Jacobian
    {
        private Matrix<double> J;

        private Matrix<double>[] dA;
        private Matrix<double>[] delta;
        private Matrix<double>[] w;
        private Matrix<double> delta1W1;
        private Matrix<double> neg1Matrix;

        private readonly IVectorSet input;
        private readonly MLPNetwork network;

        public Jacobian(MLPNetwork network, IVectorSet input)
        {
            this.input = input;
            this.network = network;

            network.StructureChanged += NetworkOnStructureChanged;
            J = Matrix<double>.Build.Dense(network.Layers[^1].NeuronsCount * input.Count, network.TotalSynapses + network.TotalBiases);
            dA = new Matrix<double>[network.Layers.Count];
            delta = new Matrix<double>[network.Layers.Count];
            w = new Matrix<double>[network.Layers.Count];
            for (int i = 0; i < network.Layers.Count; i++)
            {
                dA[i] = Matrix<double>.Build.Dense(network.Layers[i].NeuronsCount,1);
                delta[i] = Matrix<double>.Build.Dense(network.Layers[i].NeuronsCount,1);
                w[i] = Matrix<double>.Build.Dense(network.Layers[i].Weights.RowCount, network.Layers[i].Weights.ColumnCount);
            }

            neg1Matrix = Matrix<double>.Build.Dense(network.Layers[^1].NeuronsCount, 1, -1);
        }

        private void NetworkOnStructureChanged(INetwork obj)
        {
            J = Matrix<double>.Build.Dense(network.Layers[^1].NeuronsCount * input.Count, network.TotalSynapses + network.TotalBiases);
            dA = new Matrix<double>[network.Layers.Count];
        }

        public Matrix<double> CalcJacobian()
        {
            for (int a = 0; a < input.Count; a++)
            {
                int col = 0;
                network.CalculateOutput(input[a]);

                delta1W1 = neg1Matrix;
                for (var i = network.Layers.Count - 1; i >= 0; --i)
                {
                    network.Layers[i].ActivationFunction.Derivative(network.Layers[i].Net!, dA[i]);
                    delta1W1.PointwiseMultiply(dA[i], delta[i]);

                    delta1W1 = network.Layers[i].Weights.TransposeThisAndMultiply(delta[i]);

                    var b = delta[i];
                    delta[i].TransposeAndMultiply(i > 0 ? network.Layers[i - 1].Output! : input[a], w[i]);

                    for (int j = 0; j < w[i].ColumnCount; j++)
                    {
                        for (int k = 0; k < w[i].RowCount; k++)
                        {
                            J.At(a, col++, w[i].At(k, j));
                        }
                    }

                    for (int j = 0; j < b.RowCount; j++)
                    {
                        J.At(a, col++, b.At(j, 0));
                    }
                }
            }

            return J;
        }
    }
}