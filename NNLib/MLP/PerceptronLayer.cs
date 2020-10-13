using System;
using System.Diagnostics;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;

namespace NNLib
{
    

    public class PerceptronLayer : Layer
    {
        private IActivationFunction _activationFunction;

        public PerceptronLayer(int inputsCount, int neuronsCount, IActivationFunction activationFunction, MatrixBuilder? matrixBuilder = null)
            : base(inputsCount, neuronsCount, matrixBuilder ?? new NormDistMatrixBuilder())
        {
            Guards._GtZero(inputsCount).GtZero(neuronsCount).NotNull(activationFunction);

            _activationFunction = activationFunction;
        }

        private PerceptronLayer(Matrix<double> weights, Matrix<double> biases, Matrix<double>? output,
            IActivationFunction activationFunction, MatrixBuilder? matrixBuilder = null) : base(weights, biases, output, matrixBuilder ?? new NormDistMatrixBuilder())
        {
            _activationFunction = activationFunction;
        }

        public IActivationFunction ActivationFunction
        {
            get => _activationFunction;
            set
            {
                Guards._NotNull(value);
                _activationFunction = value;
            }
        }


        internal PerceptronLayer Clone() =>
            new PerceptronLayer(Weights.Clone(), Biases.Clone(), Output?.Clone(), ActivationFunction);


        public override void CalculateOutput(Matrix<double> input)
        {
            Output = Weights.Multiply(input);

            if (input.ColumnCount == 1)
            {
                Output.Add(Biases, Output);
            }
            else
            {
                Output.Add(Biases * Matrix<double>.Build.Dense(1, Output.ColumnCount, 1), Output);
            }

            Output = ActivationFunction.Function(Output);
        }
    }
}