using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using MathNet.Numerics.LinearAlgebra;
using NNLib.Common;

namespace NNLib
{
    public class LevenbergMarquardtAlgorithm : AlgorithmBase
    {
        private const double MinDampingParameter = 1.0e-6d;
        private const double MaxDampingParameter = 1.0e+6d;
        private SupervisedSet _trainingData;
        private MLPNetwork _network;
        private ILossFunction _lossFunction;

        private double _previousError;
        private Matrix<double>? _previousE = null;
        private int k;
        private double _dampingParameter;

        public LevenbergMarquardtAlgorithm(LevenbergMarquardtParams parameters)
        {
            Params = parameters;
        }
        
        public LevenbergMarquardtParams Params { get; set; }

        public override void Setup(SupervisedSet trainingData, MLPNetwork network, ILossFunction lossFunction)
        {
            _lossFunction = lossFunction;
            _network = network;
            _trainingData = trainingData;
            BatchTrainer = null;

            k = 0;
            /*var max = Double.MinValue;
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

            _dampingParameter = max > MaxDampingParameter ? MaxDampingParameter : (max < MinDampingParameter ? MinDampingParameter : max);*/
            _dampingParameter = 1000;
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

        public override int Iterations => k;
        
        private void UpdateWeightsAndBiasesWithDeltaRule(LearningMethodResult result)
        {
            if (result.Weigths.Length != result.Biases.Length)
            {
                throw new Exception();
            }
        
            for (int i = 0; i < result.Weigths.Length; i++)
            {
                _network.Layers[i].Weights.Subtract(result.Weigths[i], _network.Layers[i].Weights);
                _network.Layers[i].Biases.Subtract(result.Biases[i], _network.Layers[i].Biases);
            }
        }

        private void ResetWeightsAndBiases(LearningMethodResult result)
        {
            if (result.Weigths.Length != result.Biases.Length)
            {
                throw new Exception();
            }

            for (int i = 0; i < result.Weigths.Length; i++)
            {
                _network.Layers[i].Weights.Add(result.Weigths[i], _network.Layers[i].Weights);
                _network.Layers[i].Biases.Add(result.Biases[i], _network.Layers[i].Biases);
            }
        }

        private double CalcError(Matrix<double> E)
        {
            return E.PointwisePower(2).Enumerate().Sum() / _network.Layers[^1].NeuronsCount;
        }
        
        public override bool DoIteration(in CancellationToken ct = default)
        {
            var result = LearningMethodResult.FromNetwork(_network);
            double error;
            do
            {
                if(ct.IsCancellationRequested) throw new TrainingCanceledException();
                
                var E = _previousE ?? CalcE();

                var J = JacobianApproximation.CalcJacobian(_network, _lossFunction, _trainingData, E);
                var Jt = J.Transpose();
                var g = Jt * E;
                var JtJ = Jt * J;
                var diag = Matrix<double>.Build.Dense(JtJ.RowCount, JtJ.ColumnCount, 0);
                diag.SetDiagonal(JtJ.Diagonal());
                var G = JtJ + _dampingParameter * diag;
                //todo infinity exc
                var d = G.PseudoInverse() * g;
                var delta = d.RowSums();

                if (double.IsNaN(delta.Sum()))
                {
                    throw new AlgorithmFailed("delta contains NaN");
                }

                SetResults(result, delta, _network);
                UpdateWeightsAndBiasesWithDeltaRule(result);


                E = CalcE();
                error = CalcError(E);
                _previousE = E;

                if (k >= 1)
                {
                    if (error >= _previousError)
                    {
                        _dampingParameter *= Params.DampingParamIncFactor;
                        if (_dampingParameter > MaxDampingParameter) _dampingParameter = new Random().NextDouble() + MinDampingParameter;
                        ResetWeightsAndBiases(result);
                    }
                    else
                    {
                        _dampingParameter *= Params.DampingParamDecFactor;
                        if (_dampingParameter < MinDampingParameter) _dampingParameter = new Random().NextDouble() + MinDampingParameter;
                        break;
                    }
                }
                else break;

            } while (true);
            k++;
            _previousError = error;

            return true;
        }

        private Matrix<double> CalcE()
        {
            var E = new double[_network.Layers[^1].NeuronsCount, _trainingData.Target.Count];
            for (int i = 0; i < _trainingData.Input.Count; i++)
            {
                var input = _trainingData.Input[i];
                var target = _trainingData.Target[i];
                
                _network.CalculateOutput(input);
                var y = _lossFunction.Derivative(_network.Output, target);

                for (int j = 0; j < y.RowCount; j++)
                {
                    E[j, i] = y[j, 0];

                }
            }

            return Matrix<double>.Build.DenseOfArray(E).Transpose();
        }

    }
}