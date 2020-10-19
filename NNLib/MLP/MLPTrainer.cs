using System;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using NNLib.Data;
using NNLib.Exceptions;
using NNLib.LossFunction;
using NNLib.Training;

namespace NNLib.MLP
{
    internal class LoadedSupervisedTrainingData
    {
        public LoadedSupervisedTrainingData(SupervisedTrainingData data)
        {
            (I_Train, T_Train) = data.TrainingSet.ReadAllSamples();
            if (data.ValidationSet != null)
            {
                (I_Val, T_Val) = data.ValidationSet.ReadAllSamples();
            }
            
            if (data.TestSet != null)
            {
                (I_Test, T_Test) = data.TestSet.ReadAllSamples();
            }
        }

        public Matrix<double> I_Train;
        public Matrix<double> T_Train;

        public Matrix<double>? I_Val;
        public Matrix<double>? T_Val;
        
        public Matrix<double>? I_Test;
        public Matrix<double>? T_Test;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public (Matrix<double> I, Matrix<double> T) GetSamples(DataSetType setType)
        {
            return setType switch
            {
                DataSetType.Training => (I_Train, T_Train),
                DataSetType.Test => (I_Test!, T_Test!),
                DataSetType.Validation => (I_Val!, T_Val!),
                _ => throw new NotImplementedException(),
            };
        }
    }

    public class MLPTrainer
    {
        private AlgorithmBase _algorithm;
        private SupervisedTrainingData _trainingData;
        private LoadedSupervisedTrainingData _loadedData;

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
            _loadedData = new LoadedSupervisedTrainingData(TrainingSets);
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
                _loadedData = new LoadedSupervisedTrainingData(TrainingSets);
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

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void CheckTrainingCancelationIsRequested(in CancellationToken ct)
        {
            if (ct.IsCancellationRequested)
            {
                throw new TrainingCanceledException();
            }
        }

        private double CalculateNetworkError(in CancellationToken ct, DataSetType setType)
        {
            CheckTrainingCancelationIsRequested(ct);

            var (I, T) = _loadedData.GetSamples(setType);
            Network.CalculateOutput(I);
            
            CheckTrainingCancelationIsRequested(ct);

            return LossFunction.Function(Network.Output!, T).Enumerate().Sum() / I.ColumnCount;
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
                Error = CalculateNetworkError(ct, DataSetType.Training);
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
            Error = CalculateNetworkError(ct, DataSetType.Training);


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

            return CalculateNetworkError(ct, DataSetType.Validation);
        }

        public double RunTest(in CancellationToken ct = default)
        {
            if (TrainingSets.TestSet == null)
            {
                throw new NullReferenceException("Null validation set");
            }

            return CalculateNetworkError(ct, DataSetType.Test);
        }
    }
}