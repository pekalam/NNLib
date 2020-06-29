﻿using System;
using System.Threading;
using MathNet.Numerics.LinearAlgebra;
using NNLib.Common;

namespace NNLib
{
    public class BatchTrainer
    {
        private BatchParams _parameters;
        private SupervisedSet? _trainingSet;
        private int _setIndex;
        private LearningMethodResult[]? _methodResults;

        public BatchTrainer(BatchParams parameters)
        {
            _parameters = parameters;
        }

        public SupervisedSet? TrainingSet
        {
            get => _trainingSet;
            set
            {
                Guards._NotNull(_parameters);
                ValidateParamsForSet(value);
                ResetTrainingVars();
                IterationsPerEpoch = value.Input.Count / _parameters.BatchSize;
                _methodResults = new LearningMethodResult[IterationsPerEpoch];
                _trainingSet = value;
            }
        }

        public BatchParams? Parameters
        {
            get => _parameters;
            set
            {
                Guards._NotNull(_trainingSet);
                ValidateParamsForSet(_trainingSet);
                ResetTrainingVars();
                IterationsPerEpoch = _trainingSet.Input.Count / value.BatchSize;
                _methodResults = new LearningMethodResult[IterationsPerEpoch];
                _parameters = value;
            }
        }

        private void ValidateParamsForSet(SupervisedSet set)
        {
            if (_parameters.BatchSize > set.Input.Count)
            {
                throw new ArgumentException($"Invalid batch size {_parameters.BatchSize} for training set with count {set.Input.Count}");
            }

            if (set.Input.Count % _parameters.BatchSize != 0)
            {
                //TODO
                throw new ArgumentException($"Cannot divide training set");
            }
        }


        private void CheckTrainingCancelationIsRequested(in CancellationToken ct)
        {
            if (ct.IsCancellationRequested)
            {
                throw new TrainingCanceledException();
            }
        }

        private void ResetTrainingVars()
        {
            Iterations = IterationsPerEpoch = CurrentBatch = _setIndex = 0;
        }

        public void Reset() => ResetTrainingVars();

        public int Iterations { get; private set; }
        public int IterationsPerEpoch { get; private set; }
        public int CurrentBatch { get; private set; }

        private LearningMethodResult EndEpoch()
        {
            var result = _methodResults[0];

            for (int i = 1; i < _methodResults.Length; i++)
            {
                for (int j = 0; j < result.Weigths.Length; j++)
                {
                    result.Weigths[j] = result.Weigths[j] + _methodResults[i].Weigths[j];
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

            CheckTrainingCancelationIsRequested(ct);

            var result = func(input, expected);
            _methodResults[Iterations] = result;

            _setIndex = (_setIndex + 1) % _trainingSet.Input.Count;

            CurrentBatch = ++CurrentBatch % (_trainingSet.Input.Count / _parameters.BatchSize);

            Iterations++;
            if (Iterations == IterationsPerEpoch)
            {
                Iterations = 0;
                return EndEpoch();
            }

            return null;
        }

        public LearningMethodResult DoEpoch(Func<Matrix<double>, Matrix<double>, LearningMethodResult> func, in CancellationToken ct = default)
        {
            for (int i = 0; i < IterationsPerEpoch - 1; i++)
            {
                CheckTrainingCancelationIsRequested(ct);
                DoIteration(func, ct);
            }
            CheckTrainingCancelationIsRequested(ct);
            var result = DoIteration(func, ct);
            return result;
        }
    }
}