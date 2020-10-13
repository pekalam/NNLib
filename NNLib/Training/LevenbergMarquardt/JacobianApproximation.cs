using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using NNLib.Common;

namespace NNLib
{
    public static class JacobianApproximation
    {
        private const double StepSize = 0.001;

        public static Matrix<double> CalcJacobian(MLPNetwork network, ILossFunction lossFunction, IEnumerator<Matrix<double>> inputEnum, IEnumerator<Matrix<double>> targetEnum, SupervisedSet set, Matrix<double> E)
        {
            var J = Matrix<double>.Build.Dense(network.Layers[^1].NeuronsCount * set.Target.Count, network.TotalSynapses + network.TotalBiases);

            int s = 0;
            while(!(!inputEnum.MoveNext() || !targetEnum.MoveNext()))
            {
                var input = inputEnum.Current;
                var expected = targetEnum.Current;

                int colPos = 0;
                for (int j = 0; j < network.TotalLayers; j++)
                {
                    var w = network.Layers[j].Weights;
                    for (int k = 0; k < w.ColumnCount; k++)
                    {
                        for (int l = 0; l < w.RowCount; l++)
                        {
                            var pw = w.At(l,k);
                            var del = StepSize * (1 + Math.Abs(pw));
                            w[l,k] += del;
                            network.CalculateOutput(input);
                            var y1 = lossFunction.Derivative(network.Output!, expected);

                            for (int i = 0; i < y1.RowCount; i++)
                            {
                                y1[i,0] -= E.At(s,i);
                            }
                            
                            var d = y1.Divide(del);
                            for (int i = 0; i < d.RowCount; i++)
                            {
                                J[i, colPos] = d.At(i, 0);
                            }

                            w[l,k] = pw;
                            colPos++;
                        }
                    }
                }

                for (int j = 0; j < network.TotalLayers; j++)
                {
                    var b = network.Layers[j].Biases;
                    for (int i = 0; i < b.RowCount; i++)
                    {
                        var pb = b.At(i, 0);
                        var del = StepSize * (1 + Math.Abs(pb));
                        b[i, 0] += del;
                        network.CalculateOutput(input);
                        var y1 = lossFunction.Derivative(network.Output!, expected);
                        
                        for (int k = 0; k < y1.RowCount; k++)
                        {
                            y1[k,0] -= E.At(s,k);
                        }
                        
                        var d = y1.Divide(del);
                        for (int k = 0; k < d.RowCount; k++)
                        {
                            J[k, colPos] = d.At(k, 0);
                        }

                        b[i, 0] = pb;
                        colPos++;
                    }
                }

                s++;
            }
            
            inputEnum.Reset();
            targetEnum.Reset();
            return J;
        }
    }
}