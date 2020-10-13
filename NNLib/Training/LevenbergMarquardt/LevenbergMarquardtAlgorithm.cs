using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using MathNet.Numerics.LinearAlgebra;
using NNLib.Common;

namespace NNLib
{
    public class LevenbergMarquardtAlgorithm : AlgorithmBase
    {
        private const double MinDampingParameter = 1.0e-25;
        private const double MaxDampingParameter = 1e25;
        private const int MaxIterations = 10;
        private SupervisedSet _trainingData;
        private MLPNetwork _network;
        private ILossFunction _lossFunction;
        private Matrix<double> P;
        private Matrix<double> T;

        private double _previousError;
        private Matrix<double>? _previousE = null;
        private int k;
        private double _dampingParameter = 0.1;

        private IEnumerator<Matrix<double>> _inputEnum;
        private IEnumerator<Matrix<double>> _targetEnum;

        public LevenbergMarquardtAlgorithm(LevenbergMarquardtParams parameters)
        {
            Params = parameters;
        }
        
        public LevenbergMarquardtParams Params { get; set; }

        internal override void Setup(SupervisedSet trainingData, MLPNetwork network, ILossFunction lossFunction)
        {
            _lossFunction = lossFunction;
            _network = network;
            _trainingData = trainingData;

            T = Matrix<double>.Build.Dense(_network.Layers[^1].NeuronsCount, _trainingData.Target.Count);
            P = Matrix<double>.Build.Dense(_network.Layers[0].InputsCount, _trainingData.Input.Count);
            for (int i = 0; i < trainingData.Input.Count; i++)
            {
                P.SetColumn(i, trainingData.Input[i].Column(0));
                T.SetColumn(i, trainingData.Target[i].Column(0));
            }

            k = 0;

            _inputEnum = trainingData.Input.GetEnumerator();
            _targetEnum = trainingData.Target.GetEnumerator();
        }

        private void SetDampingParameter()
        {
            var max = Double.MinValue;
            var E = _previousE ?? CalcE();

            var J = JacobianApproximation.CalcJacobian(_network, _lossFunction, _inputEnum,_targetEnum,_trainingData, E);
            var Jt = J.Transpose();
            var g = Jt * E;
            var JtJ = Jt * J;
            var m = JtJ.Evd().EigenValues.Enumerate().Max(v => v.Real);

            if (m > max)
            {
                max = m;
            }

            _dampingParameter = max > MaxDampingParameter ? MaxDampingParameter : (max < MinDampingParameter ? 2 : max);
        }

        internal override void Reset()
        {
            k = 0;
        }

        private void SetResults(LearningMethodResult result, Vector<double> delta, MLPNetwork network)
        {
            int col = 0;
            for (int i = 0; i < network.TotalLayers; i++)
            {
                result.Weights[i] = network.Layers[i].Weights.Clone();
                for (int j = 0; j < network.Layers[i].InputsCount; j++)
                {
                    for (int k = 0; k < network.Layers[i].NeuronsCount; k++)
                    {
                        result.Weights[i][k, j] = delta[col++];
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

        internal override int Iterations => k;
        
        private void UpdateWeightsAndBiasesWithDeltaRule(LearningMethodResult result)
        {
            if (result.Weights.Length != result.Biases.Length)
            {
                throw new Exception();
            }
        
            for (int i = 0; i < result.Weights.Length; i++)
            {
                _network.Layers[i].Weights.Subtract(result.Weights[i], _network.Layers[i].Weights);
                _network.Layers[i].Biases.Subtract(result.Biases[i], _network.Layers[i].Biases);
            }
        }

        private void ResetWeightsAndBiases(LearningMethodResult result)
        {
            if (result.Weights.Length != result.Biases.Length)
            {
                throw new Exception();
            }

            for (int i = 0; i < result.Weights.Length; i++)
            {
                _network.Layers[i].Weights.Add(result.Weights[i], _network.Layers[i].Weights);
                _network.Layers[i].Biases.Add(result.Biases[i], _network.Layers[i].Biases);
            }
        }

        private double CalcError(Matrix<double> E)
        {
            return E.PointwisePower(2).Enumerate().Sum(); /// (T.ColumnCount * _network.Layers[^1].NeuronsCount);
        }

        internal override bool DoIteration(in CancellationToken ct = default)
        {
            // if (k == 0)
            // {
            //     SetDampingParameter();
            // }



            var result = LearningMethodResult.FromNetwork(_network);
            double error;
            int it = 0;
            do
            {
                if(ct.IsCancellationRequested) throw new TrainingCanceledException();
                
                var E = _previousE ?? CalcE();

                var J = JacobianApproximation.CalcJacobian(_network, _lossFunction, _inputEnum, _targetEnum, _trainingData, E);

                if (ct.IsCancellationRequested) throw new TrainingCanceledException();

                var Jt = J.Transpose();
                var g = Jt * E;
                var JtJ = Jt * J;
                var diag = Matrix<double>.Build.Dense(JtJ.RowCount, JtJ.ColumnCount, 0);
                diag.SetDiagonal(JtJ.Diagonal());
                var G = JtJ + _dampingParameter * diag;
                //todo infinity exc
                var d = G.PseudoInverse() * g;
                var delta = d.RowSums();

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
                        if (_dampingParameter > MaxDampingParameter)
                        {
                            _dampingParameter = MaxDampingParameter; 
                            break;
                        }
                        ResetWeightsAndBiases(result);
                    }
                    else
                    {
                        _dampingParameter *= Params.DampingParamDecFactor;
                        if (_dampingParameter < MinDampingParameter)
                        {
                            _dampingParameter = MinDampingParameter;
                        }
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
            //var E = Matrix<double>.Build.Dense(_network.Layers[^1].NeuronsCount, _trainingData.Target.Count);
            // for (int i = 0; i < _trainingData.Input.Count; i++)
            // {
            //     var input = _trainingData.Input[i];
            //     var target = _trainingData.Target[i];
            //     
            //     _network.CalculateOutput(input);
            //     var y = _lossFunction.Derivative(_network.Output, target);
            //
            //     for (int j = 0; j < y.RowCount; j++)
            //     {
            //         E[j, i] = y[j, 0];
            //
            //     }
            // }

            _network.CalculateOutput(P);
            var E = _lossFunction.Derivative(_network.Output, T);


            return E.Transpose();
        }

    }
}