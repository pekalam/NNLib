using MathNet.Numerics.LinearAlgebra;
using NNLib.ActivationFunction;
using NNLib.Data;

namespace NNLib.MLP
{
    

    public class PerceptronLayer : Layer
    {
        private IActivationFunction _activationFunction;
        private Matrix<double>? _net;
        private Matrix<double>? _netData;
        private Matrix<double>? _biasData1;
        private Matrix<double>? _biasDataResult;

        public PerceptronLayer(int inputsCount, int neuronsCount, IActivationFunction activationFunction, MatrixBuilder? matrixBuilder = null)
            : base(inputsCount, neuronsCount, matrixBuilder ?? new DefaultNormDistMatrixBuilder())
        {
            Guards._GtZero(inputsCount).GtZero(neuronsCount).NotNull(activationFunction);

            _activationFunction = activationFunction;
        }

        private PerceptronLayer(Matrix<double> weights, Matrix<double> biases, Matrix<double>? output, Matrix<double>? net, Matrix<double>? netStorage, Matrix<double>? netDataStorage,
            IActivationFunction activationFunction, MatrixBuilder? matrixBuilder = null) : base(weights, biases, output, matrixBuilder ?? new DefaultNormDistMatrixBuilder())
        {
            _activationFunction = activationFunction;
            _netData = netDataStorage;
            _net = netStorage;
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

        protected internal override void InitializeMemory()
        {
            ActivationFunction.InitMemory(this);
            _net = Matrix<double>.Build.Dense(NeuronsCount, 1);
        }

        protected internal override void InitializeMemoryForData(SupervisedTrainingSamples data)
        {
            ActivationFunction.InitMemoryForData(this, data);
            _netData = Matrix<double>.Build.Dense(NeuronsCount, data.Input.Count);
            _biasData1 = Matrix<double>.Build.Dense(1, data.Input.Count, 1);
            _biasDataResult = Matrix<double>.Build.Dense(NeuronsCount, data.Input.Count);
        }

        public Matrix<double>? Net;

        internal PerceptronLayer Clone() =>
            new PerceptronLayer(Weights.Clone(), Biases.Clone(), Output?.Clone(), Net?.Clone(), _net?.Clone(), _netData?.Clone(), ActivationFunction.Clone(), MatrixBuilder);


        public override void CalculateOutput(Matrix<double> input)
        {

            if (input.ColumnCount == 1)
            {
                Weights.Multiply(input, _net);
                Net = _net!;
                Net.Add(Biases, Net);
            }
            else
            {
                Weights.Multiply(input, _netData);
                Net = _netData!;
                Biases.Multiply(_biasData1, _biasDataResult);
                Net.Add(_biasDataResult, Net);
            }

            Output = ActivationFunction.Function(Net);
        }
    }
}