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
        private SupervisedTrainingData _data;

        public LoadedSupervisedTrainingData(SupervisedTrainingData data)
        {
            _data = data;
            (I_Train, T_Train) = (data.TrainingSet.ReadInputSamples(), data.TrainingSet.ReadTargetSamples());
            data.TrainingSet.Input.Modified += TrainingInputOnModified;
            data.TrainingSet.Target.Modified += TrainingTargetOnModified;
            if (data.ValidationSet != null)
            {
                (I_Val, T_Val) = (data.ValidationSet.ReadInputSamples(), data.ValidationSet.ReadTargetSamples());
                data.ValidationSet.Input.Modified += ValidationInputOnModified;
                data.ValidationSet.Target.Modified += ValidationTargetOnModified;
            }

            if (data.TestSet != null)
            {
                (I_Test, T_Test) = (data.TestSet.ReadInputSamples(), data.TestSet.ReadTargetSamples());
                data.TestSet.Input.Modified += TestInputOnModified;
                data.TestSet.Target.Modified += TestTargetOnModified;
            }

        }

        private void TrainingInputOnModified()
        {
            I_Train = _data.TrainingSet.ReadInputSamples();
        }

        private void TrainingTargetOnModified()
        {
            T_Train = _data.TrainingSet.ReadTargetSamples();
        }

        private void ValidationInputOnModified()
        {
            I_Val = _data.ValidationSet!.ReadInputSamples();
        }

        private void ValidationTargetOnModified()
        {
            T_Val = _data.ValidationSet!.ReadTargetSamples();
        }

        private void TestInputOnModified()
        {
            I_Test = _data.TestSet!.ReadInputSamples();
        }

        private void TestTargetOnModified()
        {
            T_Test = _data.TestSet!.ReadTargetSamples();
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
            if (setType == DataSetType.Training)
            {
                return (I_Train, T_Train);
            }

            if(setType == DataSetType.Validation)

            {
                return (I_Val!, T_Val!);
            }
            if(setType == DataSetType.Test)

            {
                return (I_Test!, T_Test!);
            }

            throw new NotImplementedException();
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
            _loadedData = new LoadedSupervisedTrainingData(TrainingSets);
            _algorithm.Setup(TrainingSets.TrainingSet, _loadedData, network, lossFunction);
        }

        public ILossFunction LossFunction { get;  }
        public SupervisedTrainingData TrainingSets
        {
            get => _trainingData;
            set
            {
                ValidateNetworkAndDataSets(Network, value);
                _trainingData = value;
                _loadedData = new LoadedSupervisedTrainingData(value);
                _algorithm.Setup(value.TrainingSet, _loadedData, Network, LossFunction);
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