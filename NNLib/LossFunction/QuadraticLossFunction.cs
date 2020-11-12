using MathNet.Numerics.LinearAlgebra;
using NNLib.ActivationFunction;
using NNLib.Data;

namespace NNLib.LossFunction
{
    public class QuadraticLossFunction : ILossFunction
    {
        private Matrix<double> _f;
        private Matrix<double> _df;

        private NetDataMatrixPool _fData;
        private NetDataMatrixPool _dfData;

        public Matrix<double> Function(Matrix<double> input, Matrix<double> target)
        {
            var cols = target.ColumnCount;
            var storage = cols == 1 ? _f : _fData.Get(cols);

            target.Subtract(input, storage);
            storage.PointwiseMultiply(storage, storage);
            storage.Multiply(0.5d, storage);

            return storage;
        }

        public Matrix<double> Derivative(Matrix<double> input, Matrix<double> target)
        {
            var cols = target.ColumnCount;

            var storage = cols == 1 ? _df : _dfData.Get(cols);

            input.Subtract(target, storage);
            return storage;
        }

        public void InitializeMemory(Layer layer, SupervisedTrainingSamples data)
        {
            _f = Matrix<double>.Build.Dense(layer.NeuronsCount, 1);
            _df = Matrix<double>.Build.Dense(layer.NeuronsCount, 1);
            _fData = new NetDataMatrixPool(layer.NeuronsCount, data.Input.Count);
            _dfData = new NetDataMatrixPool(layer.NeuronsCount, data.Input.Count);
        }
    }
}