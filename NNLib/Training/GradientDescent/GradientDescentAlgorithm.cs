using MathNet.Numerics.LinearAlgebra;
using NNLib.Data;
using NNLib.Exceptions;
using NNLib.LossFunction;
using NNLib.MLP;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;

namespace NNLib.Training.GradientDescent
{
    public class GradientDescentAlgorithm : AlgorithmBase
    {
        private MLPNetwork? _network;
        private ILossFunction? _lossFunction;
        private int _iterations;
        private ParametersUpdate? _previousDelta;

        private IEnumerator<Matrix<double>> _inputEnum = null!;
        private IEnumerator<Matrix<double>> _targetEnum = null!;

        private Matrix<double> _batchIn = null!;
        private Matrix<double> _batchT = null!;
        private Matrix<double> _batchInRemainder = null!;
        private Matrix<double> _batchTRemainder = null!;
        private int _batchSize;
        private int _batchRemainder;

        public GradientDescentAlgorithm(GradientDescentParams? parameters = null)
        {
            Params = parameters ?? new GradientDescentParams();
            Guards._GtZero(Params.BatchSize);
        }

        public GradientDescentParams Params { get; set; }
        internal override int Iterations => _iterations;
        public int IterationsPerEpoch { get; private set; }
        public int BatchIterations { get; private set; }

        internal override double? GetError() => null;

        internal override void Setup(SupervisedTrainingSamples set , LoadedSupervisedTrainingData _, MLPNetwork network, ILossFunction lossFunction)
        {
            Guards._NotNull(set).NotNull(network).NotNull(lossFunction);
            _lossFunction = lossFunction;
            _network = network;
            _iterations = 0;


            if (Params.BatchSize > set.Input.Count)
            {
                throw new ArgumentException($"Invalid batch size {Params.BatchSize} for training set with count {set.Input.Count}");
            }

            _batchSize = Params.BatchSize;
            _batchRemainder = set.Input.Count % Params.BatchSize;

            IterationsPerEpoch = set.Input.Count / Params.BatchSize;
            _batchIn = Matrix<double>.Build.Dense(set.Input[0].RowCount, Params.BatchSize);
            _batchT = Matrix<double>.Build.Dense(set.Target[0].RowCount, Params.BatchSize);
            if (_batchRemainder != 0)
            {
                IterationsPerEpoch++;
            }
            _batchInRemainder = Matrix<double>.Build.Dense(set.Input[0].RowCount, _batchRemainder);
            _batchTRemainder = Matrix<double>.Build.Dense(set.Target[0].RowCount, _batchRemainder);


            if (Params.Randomize)
            {
                (_inputEnum, _targetEnum) = RandomVectorSetEnumerator.GetInputTargetEnumerators(set.Input, set.Target);
            }
            else
            {
                _inputEnum = set.Input.GetEnumerator();
                _targetEnum = set.Target.GetEnumerator();
            }

            BatchIterations = 0;

            network.StructureChanged -= NetworkOnStructureChanged;
            network.StructureChanged += NetworkOnStructureChanged;
        }

        private void NetworkOnStructureChanged(INetwork obj)
        {
            BatchIterations = 0;
            _previousDelta = null;
        }

        internal override void Reset()
        {
            _iterations = BatchIterations = 0;
            _previousDelta = null;
        }

        private void UpdateWeightsAndBiases(ParametersUpdate delta, in CancellationToken ct)
        {
            TrainingCanceledException.ThrowIfCancellationRequested(ct);

            if (Params.Momentum > 0d && _previousDelta != null)
            {
                for (int i = 0; i < delta.Weights.Length; i++)
                {
                    _previousDelta.Weights[i].Multiply(Params.Momentum, _previousDelta.Weights[i]);
                    delta.Weights[i].Add(_previousDelta.Weights[i], delta.Weights[i]);
                    _network!.Layers[i].Weights.Subtract(delta.Weights[i], _network.Layers[i].Weights);


                    _previousDelta.Biases[i].Multiply(Params.Momentum, _previousDelta.Biases[i]);
                    delta.Biases[i].Add(_previousDelta.Biases[i], delta.Biases[i]);
                    _network.Layers[i].Biases.Subtract(delta.Biases[i], _network.Layers[i].Biases);
                }
            }
            else
            {
                for (int i = 0; i < delta.Weights.Length; i++)
                {
                    _network!.Layers[i].Weights.Subtract(delta.Weights[i], _network.Layers[i].Weights);
                    _network.Layers[i].Biases.Subtract(delta.Biases[i], _network.Layers[i].Biases);
                }
            }

            _previousDelta = delta;
        }

        private ParametersUpdate CalculateDelta(Matrix<double> input, Matrix<double> expected, in CancellationToken ct)
        {
            Debug.Assert(_network != null && _lossFunction != null, "Setup was not called");
            TrainingCanceledException.ThrowIfCancellationRequested(ct);

            var update = ParametersUpdate.EmptyFromNetwork(_network);
            _network.CalculateOutput(input);

            Matrix<double> delta1W1 = _lossFunction.Derivative(_network.Layers[^1].Output!, expected);
            for (var i = _network.Layers.Count - 1; i >= 0; --i)
            {
                TrainingCanceledException.ThrowIfCancellationRequested(ct);

                var dA = _network.Layers[i].ActivationFunction.Derivative(_network.Layers[i].Net!);
                var delta = delta1W1.PointwiseMultiply(dA);
                var deltaLr = delta.Multiply(Params.LearningRate);

                delta1W1 = _network.Layers[i].Weights.TransposeThisAndMultiply(delta);

                update.Biases[i] = deltaLr.RowSums().ToColumnMatrix();
                update.Weights[i] = deltaLr.TransposeAndMultiply(i > 0 ? _network.Layers[i-1].Output : input);
            }

            return update;
        }

        private (Matrix<double> batchIn, Matrix<double> batchT) ReadBatch(CancellationToken ct)
        {
            var batchIn = _batchIn;
            var batchT = _batchT;
            var sz = _batchSize;
            var inBatch = 0;

            if (BatchIterations == IterationsPerEpoch - 1 && _batchRemainder > 0)
            {
                batchIn = _batchInRemainder;
                batchT = _batchTRemainder;
                sz = _batchRemainder;
            }

            var batchInArr = batchIn.AsColumnMajorArray();
            var batchTArr = batchT.AsColumnMajorArray();
            int inputOffset = 0;
            int targetOffset = 0;
            while (inBatch != sz)
            {
                NextTrainingSample();

                TrainingCanceledException.ThrowIfCancellationRequested(ct);

                var iArr = _inputEnum.Current.AsColumnMajorArray();
                var tArr = _targetEnum.Current.AsColumnMajorArray();

                Array.Copy(iArr, 0, batchInArr, inputOffset, iArr.Length);
                Array.Copy(tArr, 0, batchTArr, targetOffset, tArr.Length);

                inputOffset += iArr.Length;
                targetOffset += tArr.Length;
                inBatch++;
            }
            
            return (batchIn, batchT);
        }

        private void NextTrainingSample()
        {
            if (!_inputEnum.MoveNext() || !_targetEnum.MoveNext())
            {
                _inputEnum.Reset();
                _targetEnum.Reset();
                _inputEnum.MoveNext();
                _targetEnum.MoveNext();
            }
        }

        internal override bool DoIteration(in CancellationToken ct = default)
        {
            if (BatchIterations == IterationsPerEpoch)
            {
                BatchIterations = 0;
            }

            Matrix<double> batchIn, batchT;
            ParametersUpdate delta;
            if (_batchSize == 1)
            {
                NextTrainingSample();
                delta = CalculateDelta(_inputEnum.Current, _targetEnum.Current, ct);
            }
            else
            {
                (batchIn, batchT) = ReadBatch(ct);
                delta = CalculateDelta(batchIn, batchT, ct);
            }
            UpdateWeightsAndBiases(delta, ct);

            _iterations++;
            BatchIterations++;

            if (BatchIterations == IterationsPerEpoch)
            {
                return true;
            }

            return false;
        }
    }
}