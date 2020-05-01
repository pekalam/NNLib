using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using MathNet.Numerics.LinearAlgebra;

namespace NNLib
{
    public abstract class Layer : IReadOnlyLayer
    {
        private IReadOnlyNetwork _network;

        internal Matrix<double> Weights;
        internal Matrix<double> Biases;
        internal Matrix<double> Output;

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


        internal void AssignNetwork(IReadOnlyNetwork network) => _network = network;

        public bool HasBiases { get; }
        public ReadMatrixWrapper ReadOutput => Output;
        public ReadMatrixWrapper ReadWeights => Weights;
        public bool IsOutputLayer => _network.ReadBaseLayers[_network.ReadBaseLayers.Count - 1] == this;
        public bool IsInputLayer => _network.ReadBaseLayers[0] == this;
        public bool Locked { get; private set; }

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

        private void CheckIsLocked() { if (Locked) throw new ObjectLockedException("Layer locked"); }

        internal virtual void Lock([CallerMemberName] string caller = "")
        {
            CheckIsLocked();
            Trace.WriteLine("Layer obj LOCKED by " + caller);
            Locked = true;
        }

        internal virtual void Unlock([CallerMemberName] string caller = "")
        {
            Trace.WriteLine("Layer obj UNLOCKED by " + caller);
            Locked = false;
        }

        protected abstract void BuildMatrices(int inputsCount, int neuronsCount);
        public abstract void CalculateOutput(Matrix<double> input);
    }
}