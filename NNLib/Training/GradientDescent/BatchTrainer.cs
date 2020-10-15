using System;
using System.Collections.Generic;
using System.Threading;
using MathNet.Numerics.LinearAlgebra;
using NNLib.Data;
using NNLib.Exceptions;

namespace NNLib.Training.GradientDescent
{
    /// <summary>
    /// Iterates over batches specified by parameter batch size. Calls training algorithm iteration method and sums parameter update results.
    /// </summary>
    public class BatchTrainer
    {
        private readonly int _batchSize;
        private readonly SupervisedSet _trainingSet;
        private readonly ParametersUpdate[] _methodResults;
        private int _iteration;

        private readonly IEnumerator<Matrix<double>> _inputEnum;
        private readonly IEnumerator<Matrix<double>> _targetEnum;

        internal BatchTrainer(int batchSize, SupervisedSet trainingSet, bool randomize = false)
        {
            _batchSize = batchSize;
            Guards._GtZero(_batchSize);
            ValidateParamsForSet(trainingSet);
            IterationsPerEpoch = trainingSet.Input.Count / _batchSize;
            _methodResults = new ParametersUpdate[IterationsPerEpoch];
            _trainingSet = trainingSet;

            _inputEnum = randomize ? new RandomVectorSetEnumerator(trainingSet.Input) : trainingSet.Input.GetEnumerator();
            _targetEnum = randomize ? new RandomVectorSetEnumerator(trainingSet.Target) : trainingSet.Target.GetEnumerator();
        }

        public int IterationsPerEpoch { get; }
        public int CurrentBatch { get; private set; }

        private void ValidateParamsForSet(SupervisedSet set)
        {
            if (_batchSize > set.Input.Count)
            {
                throw new ArgumentException($"Invalid batch size {_batchSize} for training set with count {set.Input.Count}");
            }

            if (set.Input.Count % _batchSize != 0)
            {
                throw new ArgumentException($"Cannot divide training set");
            }
        }

        private void CheckTrainingCancellationIsRequested(in CancellationToken ct)
        {
            if (ct.IsCancellationRequested)
            {
                throw new TrainingCanceledException();
            }
        }

        private ParametersUpdate EndEpoch()
        {
            var result = _methodResults[0];

            for (int i = 1; i < _methodResults.Length; i++)
            {
                for (int j = 0; j < result.Weights.Length; j++)
                {
                    result.Weights[j].Add(_methodResults[i].Weights[j], result.Weights[j]);
                }

                for (int j = 0; j < result.Biases.Length; j++)
                {
                    result.Biases[j].Add(_methodResults[i].Biases[j], result.Biases[j]);
                }
            }

            return result;
        }

        /// <summary>
        /// Called by algorithm. Returns training vectors from current item in batch.
        /// </summary>
        /// <param name="func">Callback in which training vectors are returned to caller. Method must return update of parameters in order to sum next updates.</param>
        /// <returns>Returns sum of parameter updates.</returns>
        internal ParametersUpdate? DoIteration(Func<Matrix<double>, Matrix<double>, ParametersUpdate> func, in CancellationToken ct = default)
        {
            if (!_inputEnum.MoveNext() || !_targetEnum.MoveNext())
            {
                _inputEnum.Reset();
                _targetEnum.Reset();
            }

            CheckTrainingCancellationIsRequested(ct);

            var result = func(_inputEnum.Current, _targetEnum.Current);
            _methodResults[_iteration] = result;

            CurrentBatch = ++CurrentBatch % (_trainingSet.Input.Count / _batchSize);

            _iteration++;
            if (_iteration == IterationsPerEpoch)
            {
                _iteration = 0;
                return EndEpoch();
            }

            return null;
        }

        internal void Reset()
        {
            _iteration = CurrentBatch = 0;
        }
    }
}