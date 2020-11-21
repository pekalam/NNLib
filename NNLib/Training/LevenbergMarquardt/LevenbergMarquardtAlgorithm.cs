using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using MathNet.Numerics;
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
        private MLPNetwork _network = null!;

        private LoadedSupervisedTrainingData _loadedSets = null!;
        private SupervisedTrainingSamples _set = null!;

        private double _previousError;
        private Matrix<double>? _previousE;
        private int k;
        private double _dampingParameter = 0.01;

        private Jacobian _jacobian = null!;
        private ParametersUpdate _update = null!;

        private Matrix<double> _E = null!;
        private Matrix<double> _Err = null!;
        private Matrix<double> _Et = null!;
        private Matrix<double> _g = null!;
        private Matrix<double> _JtJ = null!;
        private Matrix<double> _G = null!;
        private Matrix<double> _d = null!;
        private int[] ipiv = null!;



        public LevenbergMarquardtAlgorithm(LevenbergMarquardtParams? parameters = null)
        {
            Params = parameters ?? new LevenbergMarquardtParams();
        }
        
        public LevenbergMarquardtParams Params { get; set; }
        internal override int Iterations => k;

        internal override double? GetError() => _previousError;

        internal override void Setup(SupervisedTrainingSamples set, LoadedSupervisedTrainingData loadedSets, MLPNetwork network,
            ILossFunction lossFunction)
        {
            _loadedSets = loadedSets;
            _network = network;
            _set = set;

            k = 0;
            InitMemory(network, set);

            network.StructureChanged -= NetworkOnStructureChanged;
            network.StructureChanged += NetworkOnStructureChanged;

            if (network.TotalSynapses + network.TotalBiases <= 600 && set.Input.Count <= 600)
            {
                SetDampingParameter(default);
            }
            else
            {
                _dampingParameter = 0.1d;
            }
        }

        private void InitMemory(MLPNetwork network, SupervisedTrainingSamples set)
        {
            _previousE = null;
            _jacobian = new Jacobian(network, set.Input);
            _update = ParametersUpdate.FromNetwork(network);

            _E = Matrix<double>.Build.Dense(network.Layers[^1].NeuronsCount, set.Input.Count);
            _Et = Matrix<double>.Build.Dense(set.Input.Count, network.Layers[^1].NeuronsCount);
            _Err = Matrix<double>.Build.Dense(set.Input.Count, network.Layers[^1].NeuronsCount);
            _g = Matrix<double>.Build.Dense(network.TotalSynapses + network.TotalBiases, network.Layers[^1].NeuronsCount);
            _JtJ = Matrix<double>.Build.Dense(network.TotalSynapses + network.TotalBiases, network.TotalSynapses + network.TotalBiases);
            _G = Matrix<double>.Build.Dense(network.TotalSynapses + network.TotalBiases, network.TotalSynapses + network.TotalBiases);
            _d = Matrix<double>.Build.Dense(network.TotalSynapses + network.TotalBiases, network.Layers[^1].NeuronsCount);

            ipiv = new int[_G.RowCount];
        }

        internal override void Reset()
        {
            k = 0;
            _previousE = null;
        }

        private void NetworkOnStructureChanged(INetwork obj)
        {
            if(obj.BaseLayers[0].InputsCount != _loadedSets.I_Train.RowCount ||
               obj.BaseLayers[^1].NeuronsCount != _loadedSets.T_Train.RowCount) return;

            InitMemory(_network, _set);
        }

        private void SetDampingParameter(in CancellationToken ct)
        {
            var max = Double.MinValue;
            
            var (J, Jt) = _jacobian.CalcJacobian(ct);
   
            var JtJ = Jt * J;
            var m = JtJ.Evd().EigenValues.Enumerate().Max(v => v.Real) * 100;
            
            if (m > max)
            {
                max = m;
            }
            
            _dampingParameter = max > MaxDampingParameter ? MaxDampingParameter : (max < MinDampingParameter ? 0.1 : max);
        }

        private void PrepareUpdate(Vector<double> delta, MLPNetwork network)
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


        private void UpdateWeightsAndBiases(Vector<double> delta, MLPNetwork network)
        {
            PrepareUpdate(delta, network);
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
            return _Err.Enumerate().Sum() / (2 * E.RowCount);
        }

        internal override bool DoIteration(in CancellationToken ct = default)
        {
            void calcGInverse()
            {
                try
                {
                    LinearAlgebraControl.Provider.LUFactor((_G as DenseMatrix)!.Values, _G.RowCount, ipiv);
                    LinearAlgebraControl.Provider.LUInverseFactored((_G as DenseMatrix)!.Values, _G.RowCount, ipiv);
                }
                catch (InvalidParameterException)
                {
                    //cannot perform LU factorization
                    throw new AlgorithmFailed();
                }
            }

            var (P, T) = _loadedSets.GetSamples(DataSetType.Training);
            int it = 0;

            var e = _previousE ?? CalcE(P, T, ct);

            TrainingCanceledException.ThrowIfCancellationRequested(ct);

            var (J, Jt) = _jacobian.CalcJacobian(ct);

            Jt.Multiply(e, _g);
            Jt.Multiply(J, _JtJ);
            var diag = Matrix<double>.Build.Diagonal(_JtJ.RowCount, _JtJ.ColumnCount, _dampingParameter);
            _JtJ.Add(diag, _G);

            TrainingCanceledException.ThrowIfCancellationRequested(ct);

            calcGInverse();
            _G.Multiply(_g, _d);

            var delta = _d.RowSums();

            UpdateWeightsAndBiases(delta, _network);


            e = CalcE(P, T, ct);
            var error = CalcError(e);
            _previousE = e;


            if (k >= 1)
            {
                if (error >= _previousError)
                {
                    bool maxParamReached = false;
                    do
                    {
                        ResetWeightsAndBiases();
                        _dampingParameter *= Params.DampingParamIncFactor;
                        if (_dampingParameter > MaxDampingParameter)
                        {
                            _dampingParameter = MaxDampingParameter;
                            maxParamReached = true;
                        }

                        diag = Matrix<double>.Build.Diagonal(_JtJ.RowCount, _JtJ.ColumnCount, _dampingParameter);
                        _JtJ.Add(diag, _G);

                        TrainingCanceledException.ThrowIfCancellationRequested(ct);

                        calcGInverse();
                        _G.Multiply(_g, _d);

                        delta = _d.RowSums();

                        UpdateWeightsAndBiases(delta, _network);

                        e = CalcE(P, T, ct);
                        error = CalcError(e);

                        it++;
                    } while (it < 5 && error >= _previousError && !maxParamReached);



                }
                else
                {
                    _dampingParameter *= Params.DampingParamDecFactor;

                    if (_dampingParameter < MinDampingParameter)
                    {
                        _dampingParameter = MinDampingParameter;
                    }
                }
            }

            k++;

            _previousError = error;


            return true;
        }

        private Matrix<double> CalcE(Matrix<double> P, Matrix<double> T,in CancellationToken ct)
        {
            TrainingCanceledException.ThrowIfCancellationRequested(ct);

            _network.CalculateOutput(P);
            T.Subtract(_network.Output!, _E);
            _E.Transpose(_Et);

            TrainingCanceledException.ThrowIfCancellationRequested(ct);
            return _Et;
        }

    }
}