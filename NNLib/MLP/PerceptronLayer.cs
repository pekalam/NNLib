using MathNet.Numerics.LinearAlgebra;
using NNLib.ActivationFunction;
using NNLib.Data;

namespace NNLib.MLP
{
    

    public class PerceptronLayer : Layer
    {
        private IActivationFunction _activationFunction;
        private Matrix<double>? _net;

        private NetDataMatrixPool? _netData;
        private NetDataMatrixPool? _biasData1;
        private NetDataMatrixPool? _biasDataResult;

        private SupervisedTrainingSamples? _activationFunctionDataInit;

        public PerceptronLayer(int inputsCount, int neuronsCount, IActivationFunction activationFunction, MatrixBuilder? matrixBuilder = null)
            : base(inputsCount, neuronsCount, matrixBuilder ?? new DefaultNormDistMatrixBuilder())
        {
            Guards._GtZero(inputsCount).GtZero(neuronsCount).NotNull(activationFunction);

            _activationFunction = activationFunction;
        }

        private PerceptronLayer(Matrix<double> weights, Matrix<double> biases, Matrix<double>? output, Matrix<double>? net, Matrix<double>? netStorage,
            NetDataMatrixPool? netDataStorage, NetDataMatrixPool? biasData1, NetDataMatrixPool? biasDataResult,
            IActivationFunction activationFunction, MatrixBuilder? matrixBuilder = null) : base(weights, biases, output, matrixBuilder ?? new DefaultNormDistMatrixBuilder())
        {
            _activationFunction = activationFunction.Clone();
            _netData = netDataStorage?.Clone();
            _biasDataResult = biasDataResult?.Clone();
            _biasData1 = biasData1?.Clone();
            _net = netStorage?.Clone();
            Net = net?.Clone();
        }

        public IActivationFunction ActivationFunction
        {
            get => _activationFunction;
            set
            {
                Guards._NotNull(value);
                _activationFunction = value;
                _activationFunction.InitMemory(this);
                if (_activationFunctionDataInit != null)
                {
                    _activationFunction.InitMemoryForData(this, _activationFunctionDataInit);
                }
            }
        }

        protected internal override void InitializeMemory()
        {
            ActivationFunction.InitMemory(this);
            _net = Matrix<double>.Build.Dense(NeuronsCount, 1);
        }

        protected internal override void InitializeMemoryForData(SupervisedTrainingSamples data)
        {
            _activationFunctionDataInit = data;
            ActivationFunction.InitMemoryForData(this, data);
            _netData = new NetDataMatrixPool(NeuronsCount, data.Input.Count);
            _biasData1 = new NetDataMatrixPool(1, data.Input.Count, 1);
            _biasDataResult = new NetDataMatrixPool(NeuronsCount, data.Input.Count);
        }

        public Matrix<double>? Net;

        internal PerceptronLayer Clone() =>
            new PerceptronLayer(Weights, Biases, Output, Net, _net, _netData, _biasData1, _biasDataResult, ActivationFunction, MatrixBuilder);


        public override void CalculateOutput(Matrix<double> input)
        {
            int cols = input.ColumnCount;
            if (cols == 1)
            {
                Weights.Multiply(input, _net);
                Net = _net!;
                Net.Add(Biases, Net);
            }
            else
            {
                Weights.Multiply(input, _netData!.Get(cols));
                Net = _netData!.Get(cols);
                Biases.Multiply(_biasData1!.Get(cols), _biasDataResult!.Get(cols));
                Net.Add(_biasDataResult!.Get(cols), Net);
            }

            Output = ActivationFunction.Function(Net);
        }
    }
}