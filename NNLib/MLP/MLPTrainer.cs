using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using NNLib.Data;
using NNLib.Exceptions;
using NNLib.LossFunction;
using NNLib.Training;

namespace NNLib.MLP
{
    public class MLPTrainer
    {
        private AlgorithmBase _algorithm;
        private SupervisedTrainingData _trainingData;

        public event Action? EpochEnd;
        public event Action? IterationEnd;

        public MLPTrainer(MLPNetwork network, SupervisedTrainingData trainingData, AlgorithmBase algorithm,
            ILossFunction lossFunction)
        {
            Guards._NotNull(network).NotNull(trainingData).NotNull(lossFunction);
            ValidateNetworkAndDataSets(network, trainingData);

            Network = network;
            LossFunction = lossFunction;
            _trainingData = trainingData;
            _algorithm = algorithm;
            _algorithm.Setup(TrainingSets.TrainingSet, network, lossFunction);
        }

        public ILossFunction LossFunction { get;  }
        public SupervisedTrainingData TrainingSets
        {
            get => _trainingData;
            set
            {
                ValidateNetworkAndDataSets(Network, value);
                _trainingData = value;
                _algorithm.Setup(value.TrainingSet, Network, LossFunction);
            }
        }
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

        private void ValidateNetworkAndDataSets(MLPNetwork network, SupervisedTrainingData trainingData)
        {
            void Validate(SupervisedTrainingSamples set)
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

            Validate(trainingData.TrainingSet);
            if (trainingData.ValidationSet != null)
            {
                Validate(trainingData.ValidationSet);
            }

            if (trainingData.TestSet != null)
            {
                Validate(trainingData.TestSet);
            }
        }

        private void CheckTrainingCancelationIsRequested(in CancellationToken ct)
        {
            if (ct.IsCancellationRequested)
            {
                throw new TrainingCanceledException();
            }
        }

        private double CalculateNetworkError(in CancellationToken ct, SupervisedTrainingSamples trainingSamples)
        {
            CheckTrainingCancelationIsRequested(ct);

            var totalDelta = Matrix<double>.Build.Dense(Network.Layers.Last().NeuronsCount, 1);
            for (int i = 0; i < trainingSamples.Input.Count; ++i)
            {
                Network.CalculateOutput(trainingSamples.Input[i]);

                CheckTrainingCancelationIsRequested(ct);

                var err = LossFunction.Function(Network.Output!,
                    trainingSamples.Target[i]);
                totalDelta.Add(err, totalDelta);
            }

            var sum = totalDelta.ColumnSums().Sum() / trainingSamples.Input.Count;
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
                Error = CalculateNetworkError(ct, TrainingSets.TrainingSet);
            }
        }

        public void Reset()
        {
            Error = double.PositiveInfinity;
            Epochs = 0;
            Algorithm.Reset();
        }

        public double DoEpoch(in CancellationToken ct = default)
        {
            while (!DoIterationInternal(ct)) { }
            Epochs++;
            EpochEnd?.Invoke();
            Error = CalculateNetworkError(ct, TrainingSets.TrainingSet);


            return Error;
        }

        public Task<double> DoEpochAsync(CancellationToken ct = default)
        {
            return Task.Run(() => DoEpoch(ct), ct);
        }

        public double RunValidation(in CancellationToken ct = default)
        {
            if (TrainingSets.ValidationSet == null)
            {
                throw new NullReferenceException("Null validation set");
            }

            return CalculateNetworkError(ct, TrainingSets.ValidationSet);
        }

        public double RunTest(in CancellationToken ct = default)
        {
            if (TrainingSets.TestSet == null)
            {
                throw new NullReferenceException("Null validation set");
            }

            return CalculateNetworkError(ct, TrainingSets.TestSet);
        }
    }
}