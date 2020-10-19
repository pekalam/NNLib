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
        private readonly IVectorSet input;
        private readonly MLPNetwork network;

        public Jacobian(MLPNetwork network, IVectorSet input)
        {
            this.input = input;
            this.network = network;

            network.StructureChanged += NetworkOnStructureChanged;
            J = Matrix<double>.Build.Dense(network.Layers[^1].NeuronsCount * input.Count, network.TotalSynapses + network.TotalBiases);
        }

        private void NetworkOnStructureChanged(INetwork obj)
        {
            J = Matrix<double>.Build.Dense(network.Layers[^1].NeuronsCount * input.Count, network.TotalSynapses + network.TotalBiases);
        }

        public Matrix<double> CalcJacobian()
        {
            for (int a = 0; a < input.Count; a++)
            {
                int col = 0;
                network.CalculateOutput(input[a]);

                var delta1W1 = Matrix<double>.Build.Dense(network.Layers[^1].NeuronsCount, 1, -1);
                for (var i = network.Layers.Count - 1; i >= 0; --i)
                {
                    var dA = network.Layers[i].ActivationFunction.Derivative(network.Layers[i].Net!);
                    var delta = delta1W1.PointwiseMultiply(dA);

                    delta1W1 = network.Layers[i].Weights.TransposeThisAndMultiply(delta);

                    var b = delta;
                    var w = delta.TransposeAndMultiply(i > 0 ? network.Layers[i - 1].Output! : input[a]);

                    for (int j = 0; j < w.ColumnCount; j++)
                    {
                        for (int k = 0; k < w.RowCount; k++)
                        {
                            J.At(a, col++, w.At(k, j));
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