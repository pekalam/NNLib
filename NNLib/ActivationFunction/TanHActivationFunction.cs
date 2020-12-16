using MathNet.Numerics.LinearAlgebra;
using NNLib.Data;

namespace NNLib.ActivationFunction
{
    public class TanHActivationFunction : IActivationFunction
    {
        private MatrixColPool _fData = null!;
        private MatrixColPool _dfData = null!;

        public TanHActivationFunction()
        {
            
        }

        private TanHActivationFunction(MatrixColPool fData, MatrixColPool dfData)
        {
            _fData = fData.Clone();
            _dfData = dfData.Clone();
        }

        public Matrix<double> Function(Matrix<double> x)
        {
            var storage = _fData!.Get(x.ColumnCount);

            x.PointwiseTanh(storage);
            return storage;
        }

        public Matrix<double> Derivative(Matrix<double> x)
        {
            var storage = _dfData!.Get(x.ColumnCount);

            x.PointwiseTanh(storage);
            storage.PointwisePower(2, storage);
            storage.Negate(storage);
            storage.Add(1, storage);

            return storage;
        }

        public void InitMemory(Layer layer)
        {
            _fData = new MatrixColPool(layer.NeuronsCount, 1);
            _dfData = new MatrixColPool(layer.NeuronsCount, 1);
        }

        public void InitMemoryForData(Layer layer, SupervisedTrainingSamples data)
        {
            _fData.ClearOtherThanColumnVec();
            _dfData.ClearOtherThanColumnVec();
            _dfData.AddToPool(data.Input.Count);
            _fData.AddToPool(data.Input.Count);
        }

        public IActivationFunction Clone()
        {
            return new TanHActivationFunction(_fData,_dfData);
        }
    }
}