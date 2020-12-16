using System.Diagnostics;
using MathNet.Numerics.LinearAlgebra;
using NNLib.Data;

namespace NNLib.ActivationFunction
{
    public class LinearActivationFunction : IActivationFunction
    {
        private MatrixColPool _fData = null!;
        private MatrixColPool _dfData = null!;

        private LinearActivationFunction(MatrixColPool fData, MatrixColPool dfData)
        {
            _fData = fData.Clone();
            _dfData = dfData.Clone();
        }

        public LinearActivationFunction()
        {
        }

        public Matrix<double> Function(Matrix<double> x)
        {
            var fData = _fData.Get(x.ColumnCount);

            x.CopyTo(fData);
            return fData;
        }

        public Matrix<double> Derivative(Matrix<double> x)
        {
            return _dfData.Get(x.ColumnCount);
        }

        public void InitMemory(Layer layer)
        {
            _dfData = new MatrixColPool(layer.NeuronsCount, 1, Matrix<double>.One);
            _fData = new MatrixColPool(layer.NeuronsCount, 1);
        }

        public void InitMemoryForData(Layer layer, SupervisedTrainingSamples data)
        {
            _dfData.ClearOtherThanColumnVec();
            _fData.ClearOtherThanColumnVec();
            _dfData.AddToPool(data.Input.Count);
            _fData.AddToPool(data.Input.Count);
        }

        public IActivationFunction Clone()
        {
            return new LinearActivationFunction(_fData!, _dfData!);
        }
    }
}