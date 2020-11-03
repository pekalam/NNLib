using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra.Factorization;
using MathNet.Numerics.Providers.LinearAlgebra;
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
        private MLPNetwork _network;

        private LoadedSupervisedTrainingData _loadedSets;

        private double _previousError;
        private Matrix<double>? _previousE = null;
        private int k;
        private double _dampingParameter = 0.1;

        private Jacobian _jacobian;
        private ParametersUpdate _update;

        private Matrix<double> _E;
        private Matrix<double> _Err;
        private Matrix<double> _Et;
        private Matrix<double> _g;
        private Matrix<double> _JtJ;
        private Matrix<double> _G;
        private Matrix<double> _GInv;
        private Matrix<double> _d;
        private int[] numArray;

        private LU<double> _lu;


        public LevenbergMarquardtAlgorithm(LevenbergMarquardtParams? parameters = null)
        {
            Params = parameters ?? new LevenbergMarquardtParams();
        }
        
        public LevenbergMarquardtParams Params { get; set; }

        internal override void Setup(SupervisedTrainingSamples set, LoadedSupervisedTrainingData loadedSets, MLPNetwork network,
            ILossFunction lossFunction)
        {
            _loadedSets = loadedSets;
            _network = network;

            k = 0;
            _previousE = null;
            _jacobian = new Jacobian(network, set.Input);
            _update = ParametersUpdate.FromNetwork(network);

            _E = Matrix<double>.Build.Dense(network.Layers[^1].NeuronsCount, set.Input.Count);
            _Et = Matrix<double>.Build.Dense(set.Input.Count, network.Layers[^1].NeuronsCount);
            _Err = Matrix<double>.Build.Dense(set.Input.Count, network.Layers[^1].NeuronsCount);
            _g = Matrix<double>.Build.Dense(network.TotalSynapses + network.TotalBiases, network.Layers[^1].NeuronsCount);
            _JtJ = Matrix<double>.Build.Dense(network.TotalSynapses + network.TotalBiases, network.TotalSynapses + network.TotalBiases);
            _G = Matrix<double>.Build.Dense(network.TotalSynapses + network.TotalBiases, network.TotalSynapses + network.TotalBiases);
            _GInv = Matrix<double>.Build.Dense(network.TotalSynapses + network.TotalBiases, network.TotalSynapses + network.TotalBiases);
            _d = Matrix<double>.Build.Dense(network.TotalSynapses + network.TotalBiases, network.Layers[^1].NeuronsCount);

            numArray = new int[_G.RowCount];

            _lu = _G.LU();
                
            network.StructureChanged -= NetworkOnStructureChanged;
            network.StructureChanged += NetworkOnStructureChanged;

            if (set.Input.Count <= 400 && network.TotalSynapses + network.TotalBiases < 300)
            {
                SetDampingParameter(default);
            }
            else
            {
                _dampingParameter = 0.1;
            }
        }

        internal override void Reset()
        {
            k = 0;
            _previousE = null;
        }

        private void NetworkOnStructureChanged(INetwork obj)
        {
            _update = ParametersUpdate.FromNetwork(_network);
        }

        private void SetDampingParameter(in CancellationToken ct)
        {
            var max = Double.MinValue;
            
            var (J, Jt) = _jacobian.CalcJacobian();
   
            var JtJ = Jt * J;
            var m = JtJ.Evd().EigenValues.Enumerate().Max(v => v.Real);
            
            if (m > max)
            {
                max = m;
            }
            
            _dampingParameter = max > MaxDampingParameter ? MaxDampingParameter : (max < MinDampingParameter ? 0.1 : max);
        }

        private void SetUpdate(Vector<double> delta, MLPNetwork network)
        {
            int col = 0;
            for (int i = network.TotalLayers - 1; i >= 0; i--)
            {
                for (int j = 0; j < network.Layers[i].InputsCount; j++)
                {
                    for (int n = 0; n < network.Layers[i].NeuronsCount; n++)
                    {
                        _update.Weights[i].At(n, j, delta.At(col++));
                    }
                }
                for (int j = 0; j < network.Layers[i].NeuronsCount; j++)
                {
                    _update.Biases[i].At(j, 0, delta.At(col++));
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

        private void UpdateWeightsAndBiasesWithDeltaRule()
        {
            for (int i = 0; i < _update.Weights.Length; i++)
            {
                _network.Layers[i].Weights.Subtract(_update.Weights[i], _network.Layers[i].Weights);
                _network.Layers[i].Biases.Subtract(_update.Biases[i], _network.Layers[i].Biases);
            }
        }

        private void ResetWeightsAndBiases()
        {
            for (int i = 0; i < _update.Weights.Length; i++)
            {
                _network.Layers[i].Weights.Add(_update.Weights[i], _network.Layers[i].Weights);
                _network.Layers[i].Biases.Add(_update.Biases[i], _network.Layers[i].Biases);
            }
        }

        private double CalcError(Matrix<double> E)
        {
            E.PointwiseMultiply(E,_Err);
            _Err.Divide(2, _Err);
            return _Err.Enumerate().Sum() / E.RowCount; /// (T.ColumnCount * _network.Layers[^1].NeuronsCount);
        }

        private Stopwatch _stw = new Stopwatch();
        public static TimeSpan Total;
        public static int TotalIt;
        internal override bool DoIteration(in CancellationToken ct = default)
        {
            var (P, T) = _loadedSets.GetSamples(DataSetType.Training);
            double error;
            int it = 0;
            do
            {
                _stw.Restart();
                var E = _previousE ?? CalcE(P,T,ct);

                CheckTrainingCancelationIsRequested(ct);

                var (J, Jt) = _jacobian.CalcJacobian();

                Jt.Multiply(E, _g);
                Jt.Multiply(J, _JtJ);
                var diag = Matrix<double>.Build.Diagonal(_JtJ.RowCount, _JtJ.ColumnCount, _dampingParameter);
                _JtJ.Add(diag, _G);

                CheckTrainingCancelationIsRequested(ct);

                LinearAlgebraControl.Provider.LUFactor((_G as DenseMatrix)!.Values, _G.RowCount, numArray);
                LinearAlgebraControl.Provider.LUInverseFactored((_G as DenseMatrix)!.Values, _G.RowCount, numArray);
                _G.Multiply(_g, _d);
                
                var delta = _d.RowSums();

                SetUpdate(delta, _network);
                UpdateWeightsAndBiasesWithDeltaRule();


                E = CalcE(P, T, ct);
                error = CalcError(E);
                _previousE = E;

                if (k >= 1 && it <= 5)
                {
                    it++;

                    if (error >= _previousError)
                    {
                        _dampingParameter *= Params.DampingParamIncFactor;
                        if (_dampingParameter > MaxDampingParameter)
                        {
                            _dampingParameter = MaxDampingParameter; 
                            break;
                        }
                        ResetWeightsAndBiases();
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
                _stw.Stop();
                Total += _stw.Elapsed;
                TotalIt++;
                Console.WriteLine("Elapsed: " + _stw.Elapsed);
            } while (true);

            if (_stw.IsRunning)
            {
                _stw.Stop();
                Total += _stw.Elapsed;
                TotalIt++;
                Console.WriteLine("Elapsed: " + _stw.Elapsed);
            }

            k++;

            _previousError = error;


            return true;
        }

        private Matrix<double> CalcE(Matrix<double> P, Matrix<double> T,in CancellationToken ct)
        {
            CheckTrainingCancelationIsRequested(ct);

            _network.CalculateOutput(P);
            T.Subtract(_network.Output!, _E);
            _E.Transpose(_Et);

            CheckTrainingCancelationIsRequested(ct);
            return _Et;
        }

    }
}