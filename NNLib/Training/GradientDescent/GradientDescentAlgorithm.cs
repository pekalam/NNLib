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
    public partial class GradientDescentAlgorithm : AlgorithmBase
    {
        private MLPNetwork? _network;
        private ILossFunction? _lossFunction;
        private int _iterations;
        private ParametersUpdate? _previousDelta;
        private ParametersUpdate[] _delta = null!;
        private int _inBatch;

        private IEnumerator<Matrix<double>> _inputEnum = null!;
        private IEnumerator<Matrix<double>> _targetEnum = null!;

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

            IterationsPerEpoch = set.Input.Count / Params.BatchSize;
            if (set.Input.Count % Params.BatchSize != 0)
            {
                IterationsPerEpoch++;
            }

            _delta = new ParametersUpdate[Params.BatchSize];

            if (Params.Randomize)
            {
                (_inputEnum, _targetEnum) = RandomVectorSetEnumerator.GetInputTargetEnumerators(set.Input, set.Target);
            }
            else
            {
                _inputEnum = set.Input.GetEnumerator();
                _targetEnum = set.Target.GetEnumerator();
            }

            _inBatch = BatchIterations = 0;

            network.StructureChanged -= NetworkOnStructureChanged;
            network.StructureChanged += NetworkOnStructureChanged;
        }

        private void NetworkOnStructureChanged(INetwork obj)
        {
            _inBatch = BatchIterations = 0;
            _previousDelta = null;
        }

        internal override void Reset()
        {
            _iterations = _inBatch = BatchIterations = 0;
            _previousDelta = null;
        }

        private ParametersUpdate DeltaAvg()
        {
            var result = _delta[0];

            for (int i = 1; i < _inBatch; i++)
            {
                for (int j = 0; j < result.Weights.Length; j++)
                {
                    result.Weights[j].Add(_delta[i].Weights[j], result.Weights[j]);
                }

                for (int j = 0; j < result.Biases.Length; j++)
                {
                    result.Biases[j].Add(_delta[i].Biases[j], result.Biases[j]);
                }
            }

            for (int j = 0; j < result.Weights.Length; j++)
            {
                result.Weights[j].Divide(Params.BatchSize, result.Weights[j]);
            }

            for (int j = 0; j < result.Biases.Length; j++)
            {
                result.Biases[j].Divide(Params.BatchSize, result.Biases[j]);
            }

            return result;
        }

        private void UpdateWeightsAndBiases(in CancellationToken ct)
        {
            var delta = DeltaAvg();

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

        private ParametersUpdate CalculateDelta(Matrix<double> input, Matrix<double> expected)
        {
            Debug.Assert(_network != null && _lossFunction != null, "Setup was not called");

            var update = ParametersUpdate.FromNetwork(_network);
            _network.CalculateOutput(input);

            Matrix<double> delta1W1 = _lossFunction.Derivative(_network.Layers[^1].Output!, expected);
            for (var i = _network.Layers.Count - 1; i >= 0; --i)
            {
                var dA = _network.Layers[i].ActivationFunction.Derivative(_network.Layers[i].Net!);
                var delta = delta1W1.PointwiseMultiply(dA);
                var deltaLr = delta.Multiply(Params.LearningRate);

                delta1W1 = _network.Layers[i].Weights.TransposeThisAndMultiply(delta);

                update.Biases[i] = deltaLr;
                update.Weights[i] = deltaLr.TransposeAndMultiply(i > 0 ? _network.Layers[i-1].Output : input);
            }

            return update;
        }


        internal override bool DoIteration(in CancellationToken ct = default)
        {
            if (BatchIterations == IterationsPerEpoch)
            {
                BatchIterations = 0;
            }

            while (_inBatch != Params.BatchSize)
            {
                if (!NextTrainingSample() && BatchIterations > 0)
                {
                    break;
                }

                TrainingCanceledException.ThrowIfCancellationRequested(ct);

                _delta[_inBatch++] = CalculateDelta(_inputEnum.Current, _targetEnum.Current);
            }
            UpdateWeightsAndBiases(ct);

            _inBatch = 0;
            _iterations++;
            BatchIterations++;

            if (BatchIterations == IterationsPerEpoch)
            {
                return true;
            }

            return false;
        }

        private bool NextTrainingSample()
        {
            if (!_inputEnum.MoveNext() || !_targetEnum.MoveNext())
            {
                _inputEnum.Reset();
                _targetEnum.Reset();
                _inputEnum.MoveNext();
                _targetEnum.MoveNext();
                return false;
            }

            return true;
        }
    }
}