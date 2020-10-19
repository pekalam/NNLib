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
        private const double MaxDampingParameter = 1.0e25;
        private SupervisedTrainingSamples _trainingData;
        private MLPNetwork _network;
        private ILossFunction _lossFunction;
        private Matrix<double> P = null!;
        private Matrix<double> T = null!;

        private double _previousError;
        private Matrix<double>? _previousE = null;
        private int k;
        private double _dampingParameter = 0.1;

        private Jacobian _jacobian;

        public LevenbergMarquardtAlgorithm(LevenbergMarquardtParams? parameters = null)
        {
            Params = parameters ?? new LevenbergMarquardtParams();
        }
        
        public LevenbergMarquardtParams Params { get; set; }

        internal override void Setup(SupervisedTrainingSamples set, MLPNetwork network, ILossFunction lossFunction)
        {
            _lossFunction = lossFunction;
            _network = network;
            _trainingData = set;

            (P, T) = set.ReadAllSamples();
            k = 0;
            _previousE = null;
            _jacobian = new Jacobian(network, set.Input);

            SetDampingParameter(default);
        }

        private void SetDampingParameter(in CancellationToken ct)
        {
            var max = Double.MinValue;
            
            var J = _jacobian.CalcJacobian();
            var Jt = J.Transpose();
            var JtJ = Jt * J;
            var m = JtJ.Evd().EigenValues.Enumerate().Max(v => v.Real);
            
            if (m > max)
            {
                max = m;
            }
            
            _dampingParameter = max > MaxDampingParameter ? MaxDampingParameter : (max < MinDampingParameter ? 0.1 : max);
        }

        internal override void Reset()
        {
            k = 0;
            _previousE = null;
        }

        private void SetResults(ParametersUpdate result, Vector<double> delta, MLPNetwork network)
        {
            int col = 0;
            for (int i = network.TotalLayers - 1; i >= 0; i--)
            {
                result.Weights[i] = network.Layers[i].Weights.Clone();
                for (int j = 0; j < network.Layers[i].InputsCount; j++)
                {
                    for (int n = 0; n < network.Layers[i].NeuronsCount; n++)
                    {
                        result.Weights[i][n, j] = delta[col++];
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
            for (int i = 0; i < result.Weights.Length; i++)
            {
                _network.Layers[i].Weights.Subtract(result.Weights[i], _network.Layers[i].Weights);
                _network.Layers[i].Biases.Subtract(result.Biases[i], _network.Layers[i].Biases);
            }
        }

        private void ResetWeightsAndBiases(ParametersUpdate update)
        {
            for (int i = 0; i < update.Weights.Length; i++)
            {
                _network.Layers[i].Weights.Add(update.Weights[i], _network.Layers[i].Weights);
                _network.Layers[i].Biases.Add(update.Biases[i], _network.Layers[i].Biases);
            }
        }

        private double CalcError(Matrix<double> E)
        {
            return E.PointwisePower(2).Divide(2).Enumerate().Sum() / E.RowCount; /// (T.ColumnCount * _network.Layers[^1].NeuronsCount);
        }


        internal override bool DoIteration(in CancellationToken ct = default)
        {
            var result = ParametersUpdate.FromNetwork(_network);
            double error;
            int it = 0;
            do
            {
                var E = _previousE ?? CalcE(ct);
                if (k == 0)
                {
                    Console.WriteLine("k==0: " + CalcError(E));
                }

                CheckTrainingCancelationIsRequested(ct);

                var J = _jacobian.CalcJacobian();

                var Jt = J.Transpose();

                var g = Jt * E;
                var JtJ = Jt * J;
                var diag = Matrix<double>.Build.Diagonal(JtJ.RowCount, JtJ.ColumnCount, _dampingParameter);
                var G = JtJ + diag;
                //todo infinity exc
                CheckTrainingCancelationIsRequested(ct);

                var d = G.PseudoInverse() * g;

                var delta = d.RowSums();

                SetResults(result, delta, _network);
                UpdateWeightsAndBiasesWithDeltaRule(result);


                E = CalcE(ct);
                error = CalcError(E);
                _previousE = E;

                if (k >= 1 && it <= 5)
                {
                    it++;

                    if (error >= _previousError)
                    {
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
            var E = T - _network.Output!;

            CheckTrainingCancelationIsRequested(ct);
            return E.Transpose();
        }

    }
}