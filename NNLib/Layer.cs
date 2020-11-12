using MathNet.Numerics.LinearAlgebra;
using System;
using System.Diagnostics;
using NNLib.Data;
using NNLib.MLP;

namespace NNLib
{
    public abstract class Layer
    {
        private readonly int _initInputsCount;
        private readonly int _initNeuronsCount;

        internal INetwork? Network;

        internal event Action<Layer>? NeuronsCountChanging;
        internal event Action<Layer>? InputsCountChanging;
        internal event Action<Layer>? NeuronsCountChanged;
        internal event Action<Layer>? InputsCountChanged;

#pragma warning disable 8618
        protected Layer(Matrix<double> weights, Matrix<double> biases,
            Matrix<double>? output, MatrixBuilder matrixBuilder)
        {
            Weights = weights;
            Output = output;
            Biases = biases;

            if (matrixBuilder is NormDistMatrixBuilder n)
            {
                //temp - builder with options
                MatrixBuilder = new NormDistMatrixBuilder(n.Options);
            }
            else
            {
                MatrixBuilder = matrixBuilder;
            }

            IsInitialized = true;
        }
#pragma warning restore 8618


#pragma warning disable 8618
        protected Layer(int inputsCount, int neuronsCount, MatrixBuilder matrixBuilder)
        {
            Guards._GtZero(inputsCount).GtZero(neuronsCount);

            _initInputsCount = inputsCount;
            _initNeuronsCount = neuronsCount;

            MatrixBuilder = matrixBuilder;
        }
#pragma warning restore 8618



        public Matrix<double> Weights;
        public Matrix<double> Biases;
        public Matrix<double>? Output;

        public bool IsInitialized { get; private set; }

        public MatrixBuilder MatrixBuilder { get; set; }

        protected internal virtual void Initialize()
        {
            Debug.Assert(_initNeuronsCount > 0 && _initInputsCount > 0 && !IsInitialized);
            MatrixBuilder.BuildAllMatrices(_initNeuronsCount, _initInputsCount, this);
            IsInitialized = true;
        }

        protected internal virtual void InitializeMemory()
        {

        }

        protected internal virtual void InitializeMemoryForData(SupervisedTrainingSamples data)
        {

        }

        internal void AssignNetwork(INetwork network)
        {
            Network = network;
        }

        internal void AdjustMatSize(Layer? previous)
        {
            if (previous != null)
            {
                MatrixBuilder.AdjustMatrices(NeuronsCount, previous.NeuronsCount, this);
            }
        }

        public bool IsOutputLayer => (Network ?? throw new Exception("Network not assigned")).BaseLayers[^1] == this;

        public int NeuronsCount
        {
            get => IsInitialized ? Weights.RowCount : _initNeuronsCount;
            set
            {
                if (value <= 0)
                {
                    throw new ArgumentException($"{nameof(NeuronsCount)} cannot be lower or equal 0");
                }

                NeuronsCountChanging?.Invoke(this);
                MatrixBuilder.AdjustMatrices(value, InputsCount, this);
                NeuronsCountChanged?.Invoke(this);
            }
        }

        public int InputsCount
        {
            get => IsInitialized ? Weights.ColumnCount : _initInputsCount;
            set
            {
                if (value <= 0)
                {
                    throw new ArgumentException($"{nameof(InputsCount)} cannot be lower or equal 0");
                }

                InputsCountChanging?.Invoke(this);
                MatrixBuilder.AdjustMatrices(NeuronsCount, value, this);
                InputsCountChanged?.Invoke(this);
            }
        }

        public void ResetParameters()
        {
            MatrixBuilder.BuildAllMatrices(NeuronsCount, InputsCount, this);
        }

        public abstract void CalculateOutput(Matrix<double> input);
    }
}