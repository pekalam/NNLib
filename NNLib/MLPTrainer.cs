using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace NNLib
{
    public class MLPTrainer
    {
        //todo params
        public MLPTrainer(MLPNetwork network, SupervisedTrainingSets trainingSets, GradientDescent learningMethod, ILossFunction lossFunction)
        {
            Guards._NotNull(network).NotNull(trainingSets).NotNull(learningMethod).NotNull(lossFunction);
            ValidateNetworkAndTrainingSets(network, trainingSets);

            Network = network;
            TrainingSets = trainingSets;
            LearningMethod = learningMethod;
            LossFunction = lossFunction;

            LearningMethod.TrainingSet = trainingSets.TrainingSet;
        }

        public ILossFunction LossFunction { get; set; }
        public SupervisedTrainingSets TrainingSets { get; }
        public GradientDescent LearningMethod { get; }
        public MLPNetwork Network { get; }

        public double Error { get; private set; } = double.MaxValue;

        public event Action EpochEnd;
        public event Action IterationEnd;

        public void SetTrainingSet() => LearningMethod.TrainingSet = TrainingSets.TrainingSet;
        public void SetValidationSet() => LearningMethod.TrainingSet = TrainingSets.ValidationSet;
        public void SetTestSet() => LearningMethod.TrainingSet = TrainingSets.TestSet;

        private void ValidateNetworkAndTrainingSets(MLPNetwork network, SupervisedTrainingSets trainingSets)
        {
            void Validate(SupervisedSet set)
            {
                if (network.Layers[0].InputsCount != set.Input[0].RowCount)
                {
                    throw new Exception("Invalid network inputs count");
                }
                if (network.Layers.Last().NeuronsCount != set.Target[0].RowCount)
                {
                    throw new Exception("Invalid network inputs count");
                }
            }
            Validate(trainingSets.TrainingSet);
            if (trainingSets.ValidationSet != null)
            {
                Validate(trainingSets.ValidationSet);
            }

            if (trainingSets.TestSet != null)
            {
                Validate(trainingSets.TestSet);
            }
        }

        protected void UpdateWeightsAndBiasesWithDeltaRule(LearningMethodResult result)
        {
            if (result.Weigths.Count != result.Biases.Count)
            {
                throw new Exception();
            }

            for (int i = 0; i < result.Weigths.Count; i++)
            {
                Network.Layers[i].Weights.Subtract(result.Weigths[i], Network.Layers[i].Weights);
                Network.Layers[i].Biases.Subtract(result.Biases[i], Network.Layers[i].Biases);
            }
        }

        private void CheckTrainingCancelationIsRequested(in CancellationToken ct)
        {
            if (ct.IsCancellationRequested)
            {
                throw new TrainingCanceledException();
            }
        }

        private double CalculateNetworkError(in CancellationToken ct)
        {
            CheckTrainingCancelationIsRequested(ct);

            var totalDelta = Matrix<double>.Build.Dense(Network.Layers.Last().NeuronsCount, 1);
            for (int i = 0; i < TrainingSets.TrainingSet.Input.Count; ++i)
            {
                Network.CalculateOutput(TrainingSets.TrainingSet.Input[i]);

                CheckTrainingCancelationIsRequested(ct);

                var err = LossFunction.Function(Network.Output,
                    TrainingSets.TrainingSet.Target[i]);
                totalDelta.Add(err, totalDelta);
            }

            var sum = totalDelta.ColumnSums()[0];
            return sum;
        }

        private bool DoIterationInternal(in CancellationToken ct)
        {
            var result = LearningMethod.DoIteration(Network, LossFunction, ct);
            
            CheckTrainingCancelationIsRequested(ct);

            if (result != null)
            {
                UpdateWeightsAndBiasesWithDeltaRule(result);
                return true;
            }

            IterationEnd?.Invoke();
            return false;
        }

        public void DoIteration(in CancellationToken ct = default)
        {
            Network.Lock();

            try
            {
                DoIterationInternal(ct);
            }
            finally
            {
                Network.Unlock();
            }
        }

        public double DoEpoch(in CancellationToken ct = default)
        {
            Network.Lock();

            try
            {
                while (!DoIterationInternal(ct)) ;
                EpochEnd?.Invoke();
                Error = CalculateNetworkError(ct);
            }
            finally
            {
                Network.Unlock();
            }

            return Error;
        }

        public Task<double> DoEpochAsync(CancellationToken ct = default)
        {
            return Task.Run(() =>
            {
                return DoEpoch(ct);
            }, ct);
        }
    }
}