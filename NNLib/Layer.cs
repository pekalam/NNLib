using MathNet.Numerics.LinearAlgebra;
using System;
using NNLib.Common;

namespace NNLib
{
    public abstract class Layer
    {
        private INetwork? _network;

        internal event Action<Layer>? NeuronsCountChanging;
        internal event Action<Layer>? InputsCountChanging;
        internal event Action<Layer>? NeuronsCountChanged;
        internal event Action<Layer>? InputsCountChanged;

        protected Layer(Matrix<double> weights, Matrix<double> biases,
            Matrix<double> output)
        {
            Guards._NotNull(weights).NotNull(output);

            Weights = weights;
            Output = output;
            Biases = biases;
        }

        public Matrix<double> Weights;
        public Matrix<double> Biases;
        public Matrix<double> Output;


        internal void AssignNetwork(INetwork network) => _network = network;

        internal void AdjustMatSize(Layer? previous)
        {
            if (previous != null)
            {
                BuildMatrices(previous.NeuronsCount, NeuronsCount, false);
            }
        }

        public bool IsOutputLayer => (_network ?? throw new Exception("Network not assigned")).BaseLayers[^1] == this;
        public bool IsInputLayer => (_network ?? throw new Exception("Network not assigned")).BaseLayers[0] == this;

        public int NeuronsCount
        {
            get => Weights.RowCount;
            set
            {
                if (value <= 0)
                {
                    throw new ArgumentException($"{nameof(NeuronsCount)} cannot be lower or equal 0");
                }

                NeuronsCountChanging?.Invoke(this);
                BuildMatrices(InputsCount, value, false);
                NeuronsCountChanged?.Invoke(this);
            }
        }

        public int InputsCount
        {
            get => Weights.ColumnCount;
            set
            {
                if (value <= 0)
                {
                    throw new ArgumentException($"{nameof(InputsCount)} cannot be lower or equal 0");
                }

                InputsCountChanging?.Invoke(this);
                BuildMatrices(value, NeuronsCount, false);
                InputsCountChanged?.Invoke(this);
            }
        }

        public void RebuildMatrices()
        {
            var n = NeuronsCount;
            var i = InputsCount;
            BuildMatrices(i, n, true);
        }

        protected abstract void BuildMatrices(int inputsCount, int neuronsCount, bool rebuildAll);

        public abstract void CalculateOutput(Matrix<double> input);
    }
}