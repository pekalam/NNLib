using MathNet.Numerics.LinearAlgebra;
using NNLib.ActivationFunction;

namespace NNLib.MLP
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

        private PerceptronLayer(Matrix<double> weights, Matrix<double> biases, Matrix<double>? output, Matrix<double>? net,
            IActivationFunction activationFunction, MatrixBuilder? matrixBuilder = null) : base(weights, biases, output, matrixBuilder ?? new NormDistMatrixBuilder())
        {
            _activationFunction = activationFunction;
            Net = net;
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

        public Matrix<double>? Net;

        internal PerceptronLayer Clone() =>
            new PerceptronLayer(Weights.Clone(), Biases.Clone(), Output?.Clone(), Net?.Clone(), ActivationFunction);


        public override void CalculateOutput(Matrix<double> input)
        {
            Net = Weights.Multiply(input);

            if (input.ColumnCount == 1)
            {
                Net.Add(Biases, Net);
            }
            else
            {
                Net.Add(Biases * Matrix<double>.Build.Dense(1, Net.ColumnCount, 1), Net);
            }

            Output = ActivationFunction.Function(Net);
        }
    }
}