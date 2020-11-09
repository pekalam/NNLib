using System;
using MathNet.Numerics.LinearAlgebra;
using NNLib.Data;

namespace NNLib.ActivationFunction
{
    public class LinearActivationFunction : IActivationFunction
    {
        private Matrix<double> _f;
        private Matrix<double> _df;
        
        private Matrix<double> _fData;
        private Matrix<double> _dfData;

        private LinearActivationFunction(Matrix<double> f, Matrix<double> df, Matrix<double> fData, Matrix<double> dfData)
        {
            _f = f;
            _df = df;
            _fData = fData;
            _dfData = dfData;
        }

        public LinearActivationFunction()
        {
            
        }

        public Matrix<double> Function(Matrix<double> x)
        {
            if (x.ColumnCount == _f.ColumnCount)
            {
                x.CopyTo(_f);
                return _f;
            }

            x.CopyTo(_fData);
            return _fData;
        }

        public Matrix<double> Derivative(Matrix<double> x)
        {
            if (x.ColumnCount == _df.ColumnCount)
            {
                return _df;
            }

            return _dfData;
        }

        public void InitMemory(Layer layer)
        {
            _df = Matrix<double>.Build.Dense(layer.NeuronsCount, 1, Matrix<double>.One);
            _f = Matrix<double>.Build.Dense(layer.NeuronsCount, 1);
        }

        public void InitMemoryForData(Layer layer, SupervisedTrainingSamples data)
        {
            _dfData = Matrix<double>.Build.Dense(layer.NeuronsCount, data.Input.Count, Matrix<double>.One);
            _fData = Matrix<double>.Build.Dense(layer.NeuronsCount, data.Input.Count);
        }

        public IActivationFunction Clone()
        {
            return new LinearActivationFunction(_f, _df, _fData, _dfData);
        }
    }
}