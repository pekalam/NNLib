using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using NNLib.Common;

namespace NNLib
{
    public class MLPTrainer
    {
        private AlgorithmBase _algorithm;
        public event Action? EpochEnd;
        public event Action? IterationEnd;

        public MLPTrainer(MLPNetwork network, SupervisedTrainingSets trainingSets, AlgorithmBase algorithm,
            ILossFunction lossFunction)
        {
            Guards._NotNull(network).NotNull(trainingSets).NotNull(lossFunction);
            ValidateNetworkAndTrainingSets(network, trainingSets);

            Network = network;
            TrainingSets = trainingSets;
            LossFunction = lossFunction;
            _algorithm = algorithm;
            _algorithm.Setup(TrainingSets.TrainingSet, network, lossFunction);
        }

        public ILossFunction LossFunction { get;  }
        public SupervisedTrainingSets TrainingSets { get; }
        public MLPNetwork Network { get; }
        public AlgorithmBase Algorithm
        {
            get => _algorithm;
            set
            {
                _algorithm = value;
                _algorithm.Setup(TrainingSets.TrainingSet, Network, LossFunction);
            }
        }

        public double Error { get; private set; } = double.PositiveInfinity;
        public int Epochs { get; private set; }
        public int Iterations => Algorithm.Iterations;
        public void ResetEpochs()
        {
            Error = double.PositiveInfinity;
            Epochs = 0;
            Algorithm.ResetIterations();
        }

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
            var result = Algorithm.DoIteration(ct);

            CheckTrainingCancelationIsRequested(ct);

            IterationEnd?.Invoke();
            return result;
        }

        public void DoIteration(in CancellationToken ct = default)
        {
            if (DoIterationInternal(ct))
            {
                Epochs++;
                EpochEnd?.Invoke();
                Error = CalculateNetworkError(ct);
            }
        }

        public double DoEpoch(in CancellationToken ct = default)
        {
            while (!DoIterationInternal(ct)) { }
            Epochs++;
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