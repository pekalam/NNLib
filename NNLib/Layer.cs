using System;
using MathNet.Numerics.LinearAlgebra;

namespace NNLib
{
    public abstract class Layer : Lockable
    {
        private INetwork _network;

        //TODO public? + UI
        internal event Action<Layer> NeuronsCountChanging;
        internal event Action<Layer> InputsCountChanging;
        internal event Action<Layer> NeuronsCountChanged;
        internal event Action<Layer> InputsCountChanged;

        protected Layer(int inputsCount, int neuronsCount, Matrix<double> weights, Matrix<double> biases,
            Matrix<double> output)
        {
            Guards._NotNull(weights).NotNull(output).GtZero(inputsCount).GtZero(neuronsCount);

            Weights = weights;
            Output = output;
            Biases = biases;

            HasBiases = biases != null;
        }

        public Matrix<double> Weights;
        public Matrix<double> Biases;
        public Matrix<double> Output;


        internal void AssignNetwork(INetwork network) => _network = network;

        public bool HasBiases { get; }
        public bool IsOutputLayer => _network.BaseLayers[^1] == this;
        public bool IsInputLayer => _network.BaseLayers[0] == this;

        public int NeuronsCount
        {
            get => Weights.RowCount;
            set
            {
                CheckIsLocked();

                if (value <= 0)
                {
                    throw new ArgumentException($"{nameof(NeuronsCount)} cannot be lower or equal 0");
                }

                NeuronsCountChanging?.Invoke(this);
                BuildMatrices(InputsCount, value);
                NeuronsCountChanged?.Invoke(this);
            }
        }

        public int InputsCount
        {
            get => Weights.ColumnCount;
            set
            {
                CheckIsLocked();

                if (value <= 0)
                {
                    throw new ArgumentException($"{nameof(InputsCount)} cannot be lower or equal 0");
                }

                InputsCountChanging?.Invoke(this);
                BuildMatrices(value, NeuronsCount);
                InputsCountChanged?.Invoke(this);
            }
        }

        protected abstract void BuildMatrices(int inputsCount, int neuronsCount);

        public void RandomizeW()
        {

        }

        public void RandomizeB()
        {
        }

        public abstract void CalculateOutput(Matrix<double> input);
    }
}