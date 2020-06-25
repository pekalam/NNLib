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
    public class LevenbergMarquardtAlgorithm : AlgorithmBase
    {
        private const double MinDampingParameter = 1.0e-6d;
        private const double MaxDampingParameter = 1.0e-6d;

        private List<Matrix<double>> _prevLossFuncVal = new List<Matrix<double>>();
        private int k;
        private double _dampingParameter;

        public LevenbergMarquardtAlgorithm(LevenbergMarquardtParams parameters)
        {
            Params = parameters;
        }
        
        public LevenbergMarquardtParams Params { get; set; }

        public override void Setup(SupervisedSet trainingData, MLPNetwork network, ILossFunction lossFunction)
        {
            k = 0;
            var max = Double.MinValue;
            for (int i = 0; i < trainingData.Input.Count; i++)
            {
                var input = trainingData.Input[i];
                var target = trainingData.Target[i];
                
                network.CalculateOutput(input);
                var e = lossFunction.Derivative(network.Output, target);
                var J = JacobianApproximation.CalcJacobian(network, lossFunction, input, target);
                var Jt = J.Transpose();
                var g = Jt * e;
                var JtJ = Jt * J;
                var m = JtJ.Evd().EigenValues.Enumerate().Max(v => v.Real);

                if (m > max)
                {
                    max = m;
                }
            }
            
            _dampingParameter = max > MaxDampingParameter ? MaxDampingParameter : (max < MinDampingParameter ? MinDampingParameter : max);
        }

        private void SetResults(LearningMethodResult result, Vector<double> delta, MLPNetwork network)
        {
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
            if (k >= 1 && _dampingParameter > 0.0d)
            {
                if (_prevLossFuncVal.Last().Enumerate().Sum() < y.Enumerate().Sum())
                {
                    _dampingParameter /= Params.DampingParameterFactor;
                }
                else if (_dampingParameter < MaxDampingParameter)
                {
                    var netCpy = network.Clone();
                    var cpyResults = LearningMethodResult.FromNetwork(netCpy);
                    netCpy.CalculateOutput(input);

                    Matrix<double> y2;
                    do
                    {
                        _dampingParameter *= Params.DampingParameterFactor;

                        if(_dampingParameter > MaxDampingParameter)
                        {
                            _dampingParameter = MaxDampingParameter;
                            break;
                        }

                        var e2 = lossFunction.Derivative(netCpy.Output, expected);
                        var J2 = JacobianApproximation.CalcJacobian(netCpy, lossFunction, input, expected);
                        var Jt2 = J2.Transpose();
                        var g2 = Jt2 * e2;
                        var JtJ2 = Jt2 * J2;
                        var G2 = JtJ2 + _dampingParameter;
                        //todo infinity exc
                        var d2 = G2.PseudoInverse() * g2;
                        var delta2 = d2.RowSums();
                    
                        SetResults(cpyResults, delta2, netCpy);
                    
                        for (int i = 0; i < cpyResults.Weigths.Length; i++)
                        {
                            netCpy.Layers[i].Weights.Subtract(cpyResults.Weigths[i], netCpy.Layers[i].Weights);
                            netCpy.Layers[i].Biases.Subtract(cpyResults.Biases[i], netCpy.Layers[i].Biases);
                        }
                        netCpy.CalculateOutput(input);
                        y2 = lossFunction.Function(netCpy.Output, expected);
                    } while (_prevLossFuncVal.Last().Enumerate().Sum() > y2.Enumerate().Sum());

                }
            }
            k++;

            if (_dampingParameter < 0.0d) _dampingParameter = 0.0d;

            var e = lossFunction.Derivative(network.Output, expected);
            var J = JacobianApproximation.CalcJacobian(network, lossFunction, input, expected);
            var Jt = J.Transpose();
            var g = Jt * e;
            var JtJ = Jt * J;
            var G = JtJ + _dampingParameter;
            //todo infinity exc
            var d = G.PseudoInverse() * g;
            var delta = d.RowSums();
            
            SetResults(result, delta, network);


            return result;
        }

    }
}