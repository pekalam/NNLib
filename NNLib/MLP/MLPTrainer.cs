using NNLib.Data;
using NNLib.Exceptions;
using NNLib.LossFunction;
using NNLib.Training;
using System;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;

namespace NNLib.MLP
{
    public class MLPTrainer
    {
        private AlgorithmBase _algorithm;
        private SupervisedTrainingData _trainingData;
        private LoadedSupervisedTrainingData _loadedData;
        private ILossFunction _lossFunction;

        public event Action? EpochEnd;
        public event Action? IterationEnd;


        public MLPTrainer(MLPNetwork network, SupervisedTrainingData trainingData, AlgorithmBase algorithm,
            ILossFunction lossFunction)
        {
            Guards._NotNull(network).NotNull(trainingData).NotNull(lossFunction);
            ValidateNetworkAndDataSets(network, trainingData);

            Network = network;
            _lossFunction = lossFunction;
            _trainingData = trainingData;
            _algorithm = algorithm;
            _loadedData = new LoadedSupervisedTrainingData(TrainingSets);
            _algorithm.Setup(TrainingSets.TrainingSet, _loadedData, network, lossFunction);
            Network.InitializeMemoryForData(_trainingData.TrainingSet);
            _lossFunction.InitializeMemory(Network.Layers[^1], TrainingSets.TrainingSet);

            Network.StructureChanged += NetworkOnStructureChanged;
        }

        public ILossFunction LossFunction
        {
            get => _lossFunction;
            set
            {
                _lossFunction = value;
                _lossFunction.InitializeMemory(Network.Layers[^1], TrainingSets.TrainingSet);
            }
        }

        public SupervisedTrainingData TrainingSets
        {
            get => _trainingData;
            set
            {
                ValidateNetworkAndDataSets(Network, value);
                _trainingData = value;
                _loadedData = new LoadedSupervisedTrainingData(value);
                _algorithm.Setup(value.TrainingSet, _loadedData, Network, LossFunction);
                Network.InitializeMemoryForData(_trainingData.TrainingSet);
                LossFunction.InitializeMemory(Network.Layers[^1], TrainingSets.TrainingSet);
            }
        }
        public MLPNetwork Network { get; }
        public AlgorithmBase Algorithm
        {
            get => _algorithm;
            set
            {
                _algorithm = value;
                _algorithm.Setup(TrainingSets.TrainingSet ,_loadedData, Network, LossFunction);
            }
        }
        public double Error { get; private set; } = double.PositiveInfinity;
        public int Epochs { get; private set; }
        public int Iterations => Algorithm.Iterations;

        private void NetworkOnStructureChanged(INetwork obj)
        {
            if (obj.BaseLayers[0].InputsCount != _loadedData.I_Train.RowCount ||
                obj.BaseLayers[^1].NeuronsCount != _loadedData.T_Train.RowCount) return;

            Network.InitializeMemoryForData(_trainingData.TrainingSet);
            LossFunction.InitializeMemory(Network.Layers[^1], TrainingSets.TrainingSet);
        }

        private void ValidateNetworkAndDataSets(MLPNetwork network, SupervisedTrainingData trainingData)
        {
            void Validate(SupervisedTrainingSamples set)
            {
                if (network.Layers[0].InputsCount != set.Input[0].RowCount)
                {
                    throw new Exception("Invalid network first layer inputs count");
                }

                if (network.Layers.Last().NeuronsCount != set.Target[0].RowCount)
                {
                    throw new Exception("Invalid network output layer neurons count");
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

        private bool DoIterationInternal(in CancellationToken ct)
        {
            var result = Algorithm.DoIteration(ct);

            CheckTrainingCancelationIsRequested(ct);

            IterationEnd?.Invoke();
            return result;
        }

        public string ProgressString() => $"Epoch: {Epochs}, error: {Error}";

        public bool DoIteration(in CancellationToken ct = default)
        {
            if (DoIterationInternal(ct))
            {
                Epochs++;
                EpochEnd?.Invoke();
                Error = Algorithm.GetError() ?? LossFunction.CalculateError(Network, _loadedData.I_Train, _loadedData.T_Train,ct).Enumerate().Sum();
                return true;
            }

            return false;
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
            Error = Algorithm.GetError() ?? LossFunction.CalculateError(Network, _loadedData.I_Train, _loadedData.T_Train,ct).Enumerate().Sum();


            return Error;
        }

        public Task<double> DoEpochAsync(CancellationToken ct = default)
        {
            return Task.Run(() => DoEpoch(ct), ct);
        }

        public double TrainingError(in CancellationToken ct = default) => LossFunction.CalculateError(Network, _loadedData.I_Train, _loadedData.T_Train,ct).Enumerate().Sum();

        public double RunValidation(in CancellationToken ct = default) => RunValidation(Network, ct);

        public double RunValidation(MLPNetwork networkCopy, in CancellationToken ct = default)
        {
            if (TrainingSets.ValidationSet == null)
            {
                throw new NullReferenceException("Null validation set");
            }
            return LossFunction.CalculateError(networkCopy, _loadedData.I_Val!, _loadedData.T_Val!,ct).Enumerate().Sum();
        }

        public double RunTest(in CancellationToken ct = default) => RunTest(Network, ct);

        public double RunTest(MLPNetwork networkCopy, in CancellationToken ct = default)
        {
            if (TrainingSets.TestSet == null)
            {
                throw new NullReferenceException("Null validation set");
            }

            return LossFunction.CalculateError(networkCopy, _loadedData.I_Test!, _loadedData.T_Test!, ct).Enumerate().Sum();
        }
    }
}