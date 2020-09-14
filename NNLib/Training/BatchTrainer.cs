using System;
using System.Threading;
using MathNet.Numerics.LinearAlgebra;
using NNLib.Common;

namespace NNLib
{
    public class BatchTrainer
    {
        private readonly int _batchSize;
        private readonly SupervisedSet _trainingSet;
        private int _setIndex;
        private readonly LearningMethodResult[] _methodResults;
        private int _iteration;

        internal BatchTrainer(int batchSize, SupervisedSet trainingSet)
        {
            _batchSize = batchSize;
            Guards._GtZero(_batchSize);
            ValidateParamsForSet(trainingSet);
            IterationsPerEpoch = trainingSet.Input.Count / _batchSize;
            _methodResults = new LearningMethodResult[IterationsPerEpoch];
            _trainingSet = trainingSet;
        }

        private void ValidateParamsForSet(SupervisedSet set)
        {
            if (_batchSize > set.Input.Count)
            {
                throw new ArgumentException($"Invalid batch size {_batchSize} for training set with count {set.Input.Count}");
            }

            if (set.Input.Count % _batchSize != 0)
            {
                //TODO
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

        public int IterationsPerEpoch { get; }
        public int CurrentBatch { get; private set; }

        private LearningMethodResult EndEpoch()
        {
            var result = _methodResults[0];

            for (int i = 1; i < _methodResults.Length; i++)
            {
                for (int j = 0; j < result.Weights.Length; j++)
                {
                    result.Weights[j] = result.Weights[j] + _methodResults[i].Weights[j];
                }

                for (int j = 0; j < result.Biases.Length; j++)
                {
                    result.Biases[j] = result.Biases[j] + _methodResults[i].Biases[j];
                }
            }

            return result;
        }

        public LearningMethodResult? DoIteration(Func<Matrix<double>, Matrix<double>, LearningMethodResult> func, in CancellationToken ct = default)
        {
            var input = _trainingSet.Input[_setIndex];
            var expected = _trainingSet.Target[_setIndex];

            CheckTrainingCancellationIsRequested(ct);

            var result = func(input, expected);
            _methodResults[_iteration] = result;

            _setIndex = (_setIndex + 1) % _trainingSet.Input.Count;

            CurrentBatch = ++CurrentBatch % (_trainingSet.Input.Count / _batchSize);

            _iteration++;
            if (_iteration == IterationsPerEpoch)
            {
                _iteration = 0;
                return EndEpoch();
            }

            return null;
        }

        public void Reset()
        {
            _iteration = CurrentBatch = 0;
        }
    }
}