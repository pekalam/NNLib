using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using MathNet.Numerics.LinearAlgebra;
using NNLib.Data;
using NNLib.Exceptions;
using NNLib.LossFunction;
using NNLib.MLP;

namespace NNLib.Training.LevenbergMarquardt
{
    public class LevenbergMarquardtAlgorithm : AlgorithmBase
    {
        private const double MinDampingParameter = 1.0e-25;
        private const double MaxDampingParameter = 1e10;
        private const int MaxIterations = 10;
        private SupervisedTrainingSamples _trainingData;
        private MLPNetwork _network;
        private ILossFunction _lossFunction;
        private Matrix<double> P;
        private Matrix<double> T;

        private double _previousError;
        private Matrix<double>? _previousE = null;
        private int k;
        private double _dampingParameter = 10000000;

        private IEnumerator<Matrix<double>> _inputEnum;
        private IEnumerator<Matrix<double>> _targetEnum;

        public LevenbergMarquardtAlgorithm(LevenbergMarquardtParams parameters)
        {
            Params = parameters;
        }
        
        public LevenbergMarquardtParams Params { get; set; }

        internal override void Setup(SupervisedTrainingSamples set, MLPNetwork network, ILossFunction lossFunction)
        {
            _lossFunction = lossFunction;
            _network = network;
            _trainingData = set;

            (T, P) = set.ReadAllSamples();

            k = 0;

            _inputEnum = set.Input.GetEnumerator();
            _targetEnum = set.Target.GetEnumerator();
        }

        private void SetDampingParameter()
        {
            // var max = Double.MinValue;
            // var E = _previousE ?? CalcE();
            //
            // var J = JacobianApproximation.CalcJacobian(_network, _lossFunction, _inputEnum,_targetEnum,_trainingData, E);
            // var Jt = J.Transpose();
            // var g = Jt * E;
            // var JtJ = Jt * J;
            // var m = JtJ.Evd().EigenValues.Enumerate().Max(v => v.Real);
            //
            // if (m > max)
            // {
            //     max = m;
            // }
            //
            // _dampingParameter = max > MaxDampingParameter ? MaxDampingParameter : (max < MinDampingParameter ? 2 : max);
        }

        internal override void Reset()
        {
            k = 0;
        }

        private void SetResults(ParametersUpdate result, Vector<double> delta, MLPNetwork network)
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
                result.Biases[i] = network.Layers[i].Biases.Clone();
                for (int j = 0; j < network.Layers[i].NeuronsCount; j++)
                {
                    result.Biases[i][j, 0] = delta[col++];
                }
            }
        }

        internal override int Iterations => k;

        private void CheckTrainingCancelationIsRequested(in CancellationToken ct)
        {
            if (ct.IsCancellationRequested)
            {
                throw new TrainingCanceledException();
            }
        }

        private void UpdateWeightsAndBiasesWithDeltaRule(ParametersUpdate result)
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

        private void ResetWeightsAndBiases(ParametersUpdate update)
        {
            if (update.Weights.Length != update.Biases.Length)
            {
                throw new Exception();
            }

            for (int i = 0; i < update.Weights.Length; i++)
            {
                _network.Layers[i].Weights.Subtract(update.Weights[i], _network.Layers[i].Weights);
                _network.Layers[i].Biases.Subtract(update.Biases[i], _network.Layers[i].Biases);
            }
        }

        private double CalcError(Matrix<double> E)
        {
            return E.PointwiseMultiply(E).Enumerate().Sum() / E.RowCount; /// (T.ColumnCount * _network.Layers[^1].NeuronsCount);
        }

        private void CheckIsNan(Matrix<double> x)
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

        internal override bool DoIteration(in CancellationToken ct = default)
        {
            // if (k == 0)
            // {
            //     SetDampingParameter();
            // }



            var result = ParametersUpdate.FromNetwork(_network);
            double error;
            int it = 0;
            do
            {
                var E = _previousE ?? CalcE(ct);

                CheckTrainingCancelationIsRequested(ct);

                var J = JacobianApproximation.CalcJacobian(_network, _lossFunction, _inputEnum, _targetEnum, _trainingData, E);
                CheckIsNan(J);

                var Jt = J.Transpose();
                CheckIsNan(Jt);
                var g = Jt * E;
                var JtJ = Jt * J;
                var diag = Matrix<double>.Build.Diagonal(JtJ.RowCount, JtJ.ColumnCount, _dampingParameter);
                var G = JtJ + diag;
                //todo infinity exc
                CheckTrainingCancelationIsRequested(ct);
                
                // var det = G.Determinant();
                // Debug.WriteLine(det);

                Matrix<double> d;
                d = G.PseudoInverse() * g;

                // if (double.IsFinite(det))
                // {
                //   d = G.PseudoInverse() * g;
                // }
                //  else
                //  {
                //      //var w = diag.Inverse();
                //d = diag.Inverse() * g;
                //  }
                 //var d = diag.Inverse() * g;
                var delta = d.RowSums();

                SetResults(result, delta, _network);
                UpdateWeightsAndBiasesWithDeltaRule(result);


                E = CalcE(ct);
                error = CalcError(E);
                _previousE = E;

                if (k >= 1 && it <= 5)
                {
                    if (error >= _previousError)
                    {
                        it++;
                        _dampingParameter *= Params.DampingParamIncFactor;
                        Debug.WriteLine("inc: " + _dampingParameter);
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
                     
                        Debug.WriteLine(_dampingParameter);
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

        private Matrix<double> CalcE(in CancellationToken ct)
        {
            CheckTrainingCancelationIsRequested(ct);

            _network.CalculateOutput(P);
            var E = _lossFunction.Derivative(_network.Output!, T);

            CheckTrainingCancelationIsRequested(ct);
            return E.Transpose();
        }

    }
}