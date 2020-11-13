using System;
using System.Collections.Generic;
using System.Threading;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Storage;
using NNLib.Data;
using NNLib.Exceptions;
using NNLib.LossFunction;
using NNLib.MLP;

namespace NNLib.Training.LevenbergMarquardt
{
    public class Jacobian
    {
        private Matrix<double> J = null!;
        private Matrix<double> Jt = null!;

        private Matrix<double>[] delta = null!;
        private Matrix<double>[] w = null!;
        private Matrix<double> delta1W1 = null!;
        private Matrix<double>[] delta1W1NetStorage = null!;
        private Matrix<double> neg1Matrix = null!;

        private readonly IVectorSet input;
        private readonly MLPNetwork network;

        public Jacobian(MLPNetwork network, IVectorSet input)
        {
            this.input = input;
            this.network = network;

            network.StructureChanged += NetworkOnStructureChanged;

            InitMemory();
        }

        private void InitMemory()
        {
            J = Matrix<double>.Build.Dense(network.Layers[^1].NeuronsCount * input.Count, network.TotalSynapses + network.TotalBiases);
            Jt = Matrix<double>.Build.Dense(J.ColumnCount, J.RowCount);
            delta = new Matrix<double>[network.Layers.Count];
            delta1W1NetStorage = new Matrix<double>[network.Layers.Count];
            w = new Matrix<double>[network.Layers.Count];
            for (int i = 0; i < network.Layers.Count; i++)
            {
                delta[i] = Matrix<double>.Build.Dense(network.Layers[i].NeuronsCount, 1);
                w[i] = Matrix<double>.Build.Dense(network.Layers[i].Weights.RowCount, network.Layers[i].Weights.ColumnCount);
                delta1W1NetStorage[i] = Matrix<double>.Build.Dense(network.Layers[i].Weights.ColumnCount, delta[i].ColumnCount);
            }

            neg1Matrix = Matrix<double>.Build.Dense(network.Layers[^1].NeuronsCount, 1, -1);
        }

        private void NetworkOnStructureChanged(INetwork obj)
        {
            InitMemory();
        }

        public (Matrix<double> J, Matrix<double> Jt) CalcJacobian(in CancellationToken ct)
        {
            for (int a = 0; a < input.Count; a++)
            {
                int col = 0;
                TrainingCanceledException.ThrowIfCancellationRequested(ct);
                network.CalculateOutput(input[a]);

                delta1W1 = neg1Matrix;
                for (var i = network.Layers.Count - 1; i >= 0; --i)
                {
                    var dA = network.Layers[i].ActivationFunction.Derivative(network.Layers[i].Net!);
                    delta1W1.PointwiseMultiply(dA, delta[i]);

                    network.Layers[i].Weights.TransposeThisAndMultiply(delta[i], delta1W1NetStorage[i]);
                    delta1W1 = delta1W1NetStorage[i];

                    var b = delta[i];
                    delta[i].TransposeAndMultiply(i > 0 ? network.Layers[i - 1].Output! : input[a], w[i]);

                    var wCol = w[i].ColumnCount;
                    var rCol = w[i].RowCount;
                    var bRow = b.RowCount;
                    for (int j = 0; j < wCol; j++)
                    {
                        for (int k = 0; k < rCol; k++)
                        {
                            J.At(a, col++, w[i].At(k, j));
                        }
                    }

                    for (int j = 0; j < bRow; j++)
                    {
                        J.At(a, col++, b.At(j, 0));
                    }
                }
            }

            J.Transpose(Jt);

            return (J, Jt);
        }
    }
}