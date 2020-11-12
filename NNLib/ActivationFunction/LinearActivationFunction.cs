using System;
using MathNet.Numerics.LinearAlgebra;
using NNLib.Data;

namespace NNLib.ActivationFunction
{
    public class LinearActivationFunction : IActivationFunction
    {
        private Matrix<double> _f;
        private Matrix<double> _df;
        
        private NetDataMatrixPool? _fData;
        private NetDataMatrixPool? _dfData;

        private LinearActivationFunction(Matrix<double> f, Matrix<double> df, NetDataMatrixPool? fData, NetDataMatrixPool? dfData)
        {
            _f = f.Clone();
            _df = df.Clone();
            _fData = fData?.Clone();
            _dfData = dfData?.Clone();
        }

        public LinearActivationFunction()
        {
            
        }

        public Matrix<double> Function(Matrix<double> x)
        {
            var cols = x.ColumnCount;
            if (cols == _f.ColumnCount)
            {
                x.CopyTo(_f);
                return _f;
            }

            x.CopyTo(_fData!.Get(cols));
            return _fData.Get(cols);
        }

        public Matrix<double> Derivative(Matrix<double> x)
        {
            var cols = x.ColumnCount;
            if (cols == _df.ColumnCount)
            {
                return _df;
            }

            return _dfData!.Get(cols);
        }

        public void InitMemory(Layer layer)
        {
            _df = Matrix<double>.Build.Dense(layer.NeuronsCount, 1, Matrix<double>.One);
            _f = Matrix<double>.Build.Dense(layer.NeuronsCount, 1);
        }

        public void InitMemoryForData(Layer layer, SupervisedTrainingSamples data)
        {
            _dfData = new NetDataMatrixPool(layer.NeuronsCount, data.Input.Count, Matrix<double>.One);
            _fData = new NetDataMatrixPool(layer.NeuronsCount, data.Input.Count);
        }

        public IActivationFunction Clone()
        {
            return new LinearActivationFunction(_f, _df, _fData, _dfData);
        }
    }
}