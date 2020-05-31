using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using NNLib.ActivationFunction;
using System;

namespace NNLib
{
    public class PerceptronLayer : Layer
    {
        private IActivationFunction _activationFunction;


        public PerceptronLayer(int inputsCount, int neuronsCount, IActivationFunction activationFunction)
            : base(BuildWeightsMatrix(inputsCount, neuronsCount),
                BuildBiasesMatrix(inputsCount, neuronsCount), BuildOutputMatrix(inputsCount, neuronsCount))
        {
            Guards._GtZero(inputsCount).GtZero(neuronsCount).NotNull(activationFunction);

            ActivationFunction = activationFunction;
        }

        private PerceptronLayer(Matrix<double> weights, Matrix<double> biases, Matrix<double> output, IActivationFunction activationFunction) : base(weights, biases, output)
        {
            ActivationFunction = activationFunction;
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

        private static Matrix<double> BuildWeightsMatrix(int inputsCount, int neuronsCount) =>
            Matrix<double>.Build.Random(neuronsCount, inputsCount, new Normal());

        private static Matrix<double> BuildBiasesMatrix(int inputsCount, int neuronsCount) =>
            Matrix<double>.Build.Random(neuronsCount, 1, new Normal());

        private static Matrix<double> BuildOutputMatrix(int inputsCount, int neuronsCount) =>
            Matrix<double>.Build.Dense(neuronsCount, 1);

        internal PerceptronLayer Clone() => new PerceptronLayer(Weights.Clone(), Biases.Clone(), Output.Clone(), ActivationFunction);

        protected override void BuildMatrices(int inputsCount, int neuronsCount)
        {
            Weights = BuildWeightsMatrix(inputsCount, neuronsCount);
            Biases = BuildBiasesMatrix(inputsCount, neuronsCount);
            Output = BuildOutputMatrix(inputsCount, neuronsCount);
        }

        public override void CalculateOutput(Matrix<double> input)
        {
            if (input.ColumnCount != 1)
            {
                throw new ArgumentException();
            }

            if (input.RowCount != InputsCount)
            {
                throw new ArgumentException();
            }

            Weights.Multiply(input, Output);
            Output.Add(Biases, Output);
            Output = ActivationFunction.Function(Output);
        }
    }
}