using MathNet.Numerics.LinearAlgebra;
using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using NNLib.Training;

namespace NNLib
{
    public class MLPTrainer
    {
        public event Action? EpochEnd;
        public event Action? IterationEnd;

        public MLPTrainer(MLPNetwork network, SupervisedTrainingSets trainingSets, AlgorithmBase algorithm,
            ILossFunction lossFunction)
        {
            Guards._NotNull(network).NotNull(trainingSets).NotNull(lossFunction);
            ValidateNetworkAndTrainingSets(network, trainingSets);

            Network = network;
            TrainingSets = trainingSets;
            BatchTrainer = new BatchTrainer(algorithm);
            LossFunction = lossFunction;

            BatchTrainer.TrainingSet = trainingSets.TrainingSet;
        }

        public ILossFunction LossFunction { get; set; }
        public BatchTrainer BatchTrainer { get; }
        public SupervisedTrainingSets TrainingSets { get; }
        public MLPNetwork Network { get; }

        public double Error { get; private set; } = double.MaxValue;

        public void SetTrainingSet() => BatchTrainer.TrainingSet = TrainingSets.TrainingSet;
        public void SetValidationSet() => BatchTrainer.TrainingSet = TrainingSets.ValidationSet;
        public void SetTestSet() => BatchTrainer.TrainingSet = TrainingSets.TestSet;

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
            if (result.Weigths.Length != result.Biases.Length)
            {
                throw new Exception();
            }

            for (int i = 0; i < result.Weigths.Length; i++)
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
            var result = BatchTrainer.DoIteration(Network, LossFunction, ct);

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
            DoIterationInternal(ct);
        }

        public double DoEpoch(in CancellationToken ct = default)
        {
            var result = BatchTrainer.DoEpoch(Network, LossFunction, ct);
            UpdateWeightsAndBiasesWithDeltaRule(result);
            EpochEnd?.Invoke();
            Error = CalculateNetworkError(ct);


            return Error;
        }

        public Task<double> DoEpochAsync(CancellationToken ct = default)
        {
            return Task.Run(() => { return DoEpoch(ct); }, ct);
        }
    }
}