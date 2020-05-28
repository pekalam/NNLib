using System;
using System.Threading;

namespace NNLib
{
    public class BatchTrainer
    {
        private GradientDescentParams _parameters;
        private SupervisedSet _trainingSet;
        private int _setIndex;
        private readonly GradientDescentAlgorithm _gradientDescent;
        private LearningMethodResult[] _methodResults;

        public BatchTrainer(GradientDescentAlgorithm gradientDescent)
        {
            _gradientDescent = gradientDescent;
            _parameters = gradientDescent.Params;
        }

        public SupervisedSet TrainingSet
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

        public GradientDescentParams Parameters
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

        public LearningMethodResult? DoIteration(MLPNetwork network, ILossFunction lossFunction, in CancellationToken ct = default)
        {
            var input = _trainingSet.Input[_setIndex];
            var expected = _trainingSet.Target[_setIndex];

            network.CalculateOutput(input);

            CheckTrainingCancelationIsRequested(ct);

            var result = _gradientDescent.CalculateDelta(input, expected, lossFunction);
            _methodResults[Iterations] = result;

            _setIndex++;
            _setIndex %= _trainingSet.Input.Count;

            CurrentBatch = ++CurrentBatch % (_trainingSet.Input.Count / _parameters.BatchSize);

            Iterations++;
            if (Iterations == IterationsPerEpoch)
            {
                Iterations = 0;
                return EndEpoch();
            }

            return null;
        }

        public LearningMethodResult DoEpoch(MLPNetwork network, ILossFunction lossFunction, in CancellationToken ct = default)
        {
            for (int i = 0; i < IterationsPerEpoch - 1; i++)
            {
                CheckTrainingCancelationIsRequested(ct);
                DoIteration(network, lossFunction, ct);
            }
            CheckTrainingCancelationIsRequested(ct);
            var result = DoIteration(network, lossFunction, ct);
            return result;
        }
    }
}