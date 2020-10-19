using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using NNLib.Data;
using NNLib.LossFunction;
using NNLib.MLP;

namespace NNLib.Training.LevenbergMarquardt
{
    public static class JacobianApproximation
    {
        private const double StepSize = 0.001;

        public static Matrix<double> CalcJacobian(MLPNetwork network, ILossFunction lossFunction, IEnumerator<Matrix<double>> inputEnum, IEnumerator<Matrix<double>> targetEnum, SupervisedTrainingSamples trainingSamples, Matrix<double> E)
        {
            var J = Matrix<double>.Build.Dense(network.Layers[^1].NeuronsCount * trainingSamples.Target.Count, network.TotalSynapses + network.TotalBiases);

            #region x
            // int s = 0;
            // while(!(!inputEnum.MoveNext() || !targetEnum.MoveNext()))
            // {
            //     var input = inputEnum.Current;
            //     var expected = targetEnum.Current;
            //
            //     int colPos = 0;
            //     for (int j = 0; j < network.TotalLayers; j++)
            //     {
            //         var w = network.Layers[j].Weights;
            //         for (int k = 0; k < w.ColumnCount; k++)
            //         {
            //             for (int l = 0; l < w.RowCount; l++)
            //             {
            //                 var pw = w.At(l,k);
            //                 var del = StepSize;
            //                 w[l,k] += del;
            //                 network.CalculateOutput(input);
            //                 var y1 = lossFunction.Derivative(network.Output!, expected);
            //
            //                 for (int i = 0; i < y1.RowCount; i++)
            //                 {
            //                     y1[i,0] -= E.At(s,i);
            //                 }
            //                 
            //                 var d = y1.Divide(del);
            //                 for (int i = 0; i < d.RowCount; i++)
            //                 {
            //                     J[i, colPos] = d.At(i, 0);
            //                 }
            //
            //                 w[l,k] = pw;
            //                 colPos++;
            //             }
            //         }
            //     }
            //
            //     for (int j = 0; j < network.TotalLayers; j++)
            //     {
            //         var b = network.Layers[j].Biases;
            //         for (int i = 0; i < b.RowCount; i++)
            //         {
            //             var pb = b.At(i, 0);
            //             var del = StepSize;
            //             b[i, 0] += del;
            //             network.CalculateOutput(input);
            //             var y1 = lossFunction.Derivative(network.Output!, expected);
            //             
            //             for (int k = 0; k < y1.RowCount; k++)
            //             {
            //                 y1[k,0] -= E.At(s,k);
            //             }
            //             
            //             var d = y1.Divide(del);
            //             for (int k = 0; k < d.RowCount; k++)
            //             {
            //                 J[k, colPos] = d.At(k, 0);
            //             }
            //
            //             b[i, 0] = pb;
            //             colPos++;
            //         }
            //     }
            //
            //     s++;
            // }


            #endregion

            void CheckIsNan(Matrix<double> x)
            {
                for (int i = 0; i < x.ColumnCount; i++)
                {
                    for (int j = 0; j < x.RowCount; j++)
                    {
                        if (double.IsNaN(x[j, i]))
                        {
                            //throw new NotImplementedException();
                        }
                    }
                }
            }


            int a = 0;
            while (!(!inputEnum.MoveNext() || !targetEnum.MoveNext()))
            {
                var input = inputEnum.Current;
                var target = targetEnum.Current;
            
                int col = 0;

                network.CalculateOutput(input);
                Matrix<double> delta1W1 = Matrix<double>.Build.Dense(network.Layers[^1].NeuronsCount, 1, -1);
                for (var i = network.Layers.Count - 1; i >= 0; --i)
                {
                    var dA = network.Layers[i].ActivationFunction.Derivative(network.Layers[i].Net!);
                    var delta = delta1W1.PointwiseMultiply(dA);
            
                    delta1W1 = network.Layers[i].Weights.TransposeThisAndMultiply(delta);
            
                    var b = delta;
                    var w = delta.TransposeAndMultiply(i > 0 ? network.Layers[i - 1].Output! : input);
                    
                    CheckIsNan(w);
                    CheckIsNan(b);

                    for (int j = 0; j < w.RowCount; j++)
                    {
                        for (int k = 0; k < w.ColumnCount; k++)
                        {
                            J[a, col++] = w[j, k];
                        }
                    }
                    
                    for (int j = 0; j < b.RowCount; j++)
                    {
                        for (int k = 0; k < b.ColumnCount; k++)
                        {
                            J[a, col++] = b[j, k];
                        }
                    }
                }
            
                a++;
            }


            inputEnum.Reset();
            targetEnum.Reset();
            return J;
        }
    }
}