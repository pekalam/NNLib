using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace NNLib.Training.LevenbergMarquardt
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

    public class LevenbergMarquardtAlgorithm : AlgorithmBase
    {
        private List<Matrix<double>> _prevLossFuncVal = new List<Matrix<double>>();
        private int k;
        private double _lambda;

        public LevenbergMarquardtAlgorithm(LevenbergMarquardtParams parameters)
        {
            Params = parameters;
            _lambda = Params.lambda;
        }
        
        public LevenbergMarquardtParams Params { get; set; }

        public override void Setup(SupervisedSet trainingData, MLPNetwork network)
        {
            throw new NotImplementedException();
        }

        public override LearningMethodResult CalculateDelta(MLPNetwork network, Matrix<double> input, Matrix<double> expected,
            ILossFunction lossFunction)
        {
            var result = LearningMethodResult.FromNetwork(network);
            network.CalculateOutput(input);
            var y = lossFunction.Function(network.Output, expected);

            if (y.Enumerate().Max() < Params.Eps)
            {
                //zero
                return result;
            }

            _prevLossFuncVal.Add(y);
            if (k >= 1 && _lambda > 0.0d)
            {
                if (_prevLossFuncVal.Last().Enumerate().Sum() < y.Enumerate().Sum())
                {
                    _lambda -= Params.LambdaStep;
                }
                else
                {
                    _lambda += Params.LambdaStep;
                }
            }

            if (_lambda < 0.0d) _lambda = 0.0d;

            var e = lossFunction.Derivative(network.Output, expected);

            var J = JacobianApproximation.CalcJacobian(network, lossFunction, input, expected);

            
            var Jt = J.Transpose();
            var g = Jt * e;
            var JtJ = Jt * J;


            if (k++ == 0)
            {
                var m = JtJ.Evd().EigenValues.Enumerate().Max(v => v.Real);
                _lambda = m;
            }

            var G = JtJ + _lambda;


            //todo infinity exc
            var d = G.PseudoInverse() * g;  
            
            var delta = d.RowSums();
            int col = 0;
            for (int i = 0; i < network.TotalLayers; i++)
            {
                result.Weigths[i] = network.Layers[i].Weights.Clone();
                for (int j = 0; j < network.Layers[i].NeuronsCount; j++)
                {
                    for (int k = 0; k < network.Layers[i].InputsCount; k++)
                    {
                        result.Weigths[i][j, k] = delta[col++];
                    }
                }
                
            }

            for (int i = 0; i < network.TotalLayers; i++)
            {
                result.Biases[i] = network.Layers[i].Biases.Clone();
                for (int j = 0; j < network.Layers[i].NeuronsCount; j++)
                {
                    result.Biases[i][j, 0] = delta[col++];
                }
            }


            return result;
        }

    }
}