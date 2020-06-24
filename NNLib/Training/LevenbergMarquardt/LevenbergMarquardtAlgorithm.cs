using System;
using System.Collections.Generic;
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

        public static (Matrix<double> J, Matrix<double> W) CalcJacobian(MLPNetwork network, ILossFunction lossFunction,
            Matrix<double> input, Matrix<double> expected)
        {
            network.CalculateOutput(input);
            var y = lossFunction.Derivative(network.Output, expected);
            var J = Matrix<double>.Build.Dense(expected.RowCount, network.TotalSynapses + network.TotalBiases);
            var W = J.Clone();

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
                        var y1 = network.Output;
                        var d = (y1 - y).Divide(del);
                        for (int i = 0; i < d.RowCount; i++)
                        {
                            J[i, colPos] = d[i, 0];
                            W[i, colPos] = pw;
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
                    var y1 = network.Output;
                    var d = (y1 - y).Divide(del);
                    for (int k = 0; k < d.RowCount; k++)
                    {
                        J[k, colPos] = d[k, 0];
                        W[k, colPos] = pb;
                    }

                    b[i, 0] = pb;
                    colPos++;
                }
            }


            return (J, W);
        }
    }

    public class LevenbergMarquardtAlgorithm : AlgorithmBase
    {
        private MLPNetwork _network;

        public LevenbergMarquardtAlgorithm(LevenbergMarquardtParams @params, MLPNetwork network)
        {
            Params = @params;
            _network = network;
        }
        
        public LevenbergMarquardtParams Params { get; set; }
        public override BatchParams BatchParams => Params;


        public override LearningMethodResult CalculateDelta(Matrix<double> input, Matrix<double> expected,
            ILossFunction lossFunction)
        {
            var result = LearningMethodResult.FromNetwork(_network);
            _network.CalculateOutput(input);
            var y = lossFunction.Function(_network.Output, expected);

            if (y.Enumerate().Max() < Params.Eps)
            {
                //zero
                return result;
            }

            var (J, W) = JacobianApproximation.CalcJacobian(_network, lossFunction, input, expected);
            var JtWJ = J.TransposeAndMultiply(W).Multiply(J);
            var g = JtWJ.Add(JtWJ * Params.lambda);
            var JtWdy = J.TransposeAndMultiply(W).Multiply(y);

            var d = g.PseudoInverse().Multiply(JtWdy);
            var delta = d.RowSums();
            int col = 0;
            for (int i = 0; i < _network.TotalLayers; i++)
            {
                result.Weigths[i] = _network.Layers[i].Weights.Clone();
                for (int j = 0; j < _network.Layers[i].NeuronsCount; j++)
                {
                    for (int k = 0; k < _network.Layers[i].InputsCount; k++)
                    {
                        result.Weigths[i][j, k] = delta[col++];
                    }
                }
                
            }

            for (int i = 0; i < _network.TotalLayers; i++)
            {
                result.Biases[i] = _network.Layers[i].Biases.Clone();
                for (int j = 0; j < _network.Layers[i].NeuronsCount; j++)
                {
                    result.Biases[i][j, 0] = delta[col++];
                }
            }


            return result;
        }

    }

    public class LevenbergMarquardtParams : BatchParams
    {
        /// <summary>
        /// Lambda parameter
        /// </summary>
        public double lambda { get; set; } = 100000;

        public double Eps { get; set; }
    }
}