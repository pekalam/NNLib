using System;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using NNLib.Data;

namespace NNLib.LossFunction
{
    public class RootMeanSquareLossFunction : ILossFunction
    {
        private MatrixColPool _fData = null!;
        private MatrixColPool _dfData = null!;
        private MatrixColPool _dfData2 = null!;
        private readonly double _c;

        public RootMeanSquareLossFunction(double c = 0.001)
        {
            _c = c;
        }


        public Matrix<double> Function(Matrix<double> input, Matrix<double> target)
        {
            var cols = target.ColumnCount;
            var storage = _fData.Get(cols);

            target.Subtract(input, storage);
            storage.PointwiseMultiply(storage, storage);
            storage = storage.RowSums().ToColumnMatrix();
            storage.Multiply(1d / cols, storage);

            storage.PointwiseSqrt(storage);

            return storage;
        }

        public Matrix<double> Derivative(Matrix<double> input, Matrix<double> target)
        {
            var cols = target.ColumnCount;
            var storage = _dfData.Get(cols);
            var storage2 = _dfData2.Get(cols);

            input.Subtract(target, storage);

            target.Subtract(input, storage2);
            storage2.PointwiseMultiply(storage2, storage2);
            storage2 = storage2.RowSums().ToColumnMatrix();
            storage2.Multiply(1d / cols, storage2);
            storage2.Add(_c, storage2);
            storage2.PointwiseSqrt(storage2);

            storage.Divide(storage2.Enumerate().Sum() * cols, storage);

            return storage;
        }

        public void InitializeMemory(Layer layer, SupervisedTrainingSamples data)
        {
            _fData = new MatrixColPool(layer.NeuronsCount, 1);
            _dfData = new MatrixColPool(layer.NeuronsCount, 1);
            _dfData2 = new MatrixColPool(layer.NeuronsCount, 1);
            _fData.AddToPool(data.Input.Count);
            _dfData.AddToPool(data.Input.Count);
            _dfData2.AddToPool(data.Input.Count);
        }
    }
}