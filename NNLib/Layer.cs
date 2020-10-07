﻿using MathNet.Numerics.LinearAlgebra;
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
            Matrix<double>? output, MatrixBuilder matrixBuilder)
        {
            Guards._NotNull(weights);

            Weights = weights;
            Output = output;
            Biases = biases;

            MatrixBuilder = matrixBuilder;
            matrixBuilder.SetLayer(this);
        }

#pragma warning disable 8618
        protected Layer(int inputsCount, int neuronsCount, MatrixBuilder matrixBuilder)
        {
            Guards._GtZero(inputsCount).GtZero(neuronsCount).NotNull(matrixBuilder);

            MatrixBuilder = matrixBuilder;
            matrixBuilder.SetLayer(this);
            MatrixBuilder.BuildAllMatrices(neuronsCount, inputsCount);
        }
#pragma warning restore 8618

        public MatrixBuilder MatrixBuilder { get; set; }

        public Matrix<double> Weights;
        public Matrix<double> Biases;
        public Matrix<double>? Output;


        internal void AssignNetwork(INetwork network) => _network = network;

        internal void AdjustMatSize(Layer? previous)
        {
            if (previous != null)
            {
                MatrixBuilder.AdjustMatrices(NeuronsCount, previous.NeuronsCount);
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
                MatrixBuilder.AdjustMatrices(value, InputsCount);
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
                MatrixBuilder.AdjustMatrices(NeuronsCount, value);
                InputsCountChanged?.Invoke(this);
            }
        }

        public void ResetParameters()
        {
            MatrixBuilder.BuildAllMatrices(NeuronsCount, InputsCount);
        }

        public abstract void CalculateOutput(Matrix<double> input);
    }
}