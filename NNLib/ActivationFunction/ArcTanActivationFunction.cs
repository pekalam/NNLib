﻿using MathNet.Numerics.LinearAlgebra;
using NNLib.Data;

namespace NNLib.ActivationFunction
{
    public class ArcTanActivationFunction : IActivationFunction
    {
        private Matrix<double> _f;
        private Matrix<double> _df;

        private Matrix<double> _fData;
        private Matrix<double> _dfData;

        private ArcTanActivationFunction(Matrix<double> f, Matrix<double> df, Matrix<double> fData, Matrix<double> dfData)
        {
            _f = f;
            _df = df;
            _fData = fData;
            _dfData = dfData;
        }

        public ArcTanActivationFunction()
        {
            
        }

        public Matrix<double> Function(Matrix<double> x)
        {
            Matrix<double> storage = x.ColumnCount == _f.ColumnCount ? _f : _fData;

            x.PointwiseAtan(storage);
            return storage;
        }

        public Matrix<double> Derivative(Matrix<double> x)
        {
            Matrix<double> storage = x.ColumnCount == _df.ColumnCount ? _df : _dfData;

            x.PointwisePower(2, storage);
            storage.Add(1, storage);
            storage.PointwisePower(-1, storage);
            return storage;
        }

        public void InitMemory(Layer layer)
        {
            _f = Matrix<double>.Build.Dense(layer.NeuronsCount, 1);
            _df = Matrix<double>.Build.Dense(layer.NeuronsCount, 1);
        }

        public void InitMemoryForData(Layer layer, SupervisedTrainingSamples data)
        {
            _dfData = Matrix<double>.Build.Dense(layer.NeuronsCount, data.Input.Count);
            _fData = Matrix<double>.Build.Dense(layer.NeuronsCount, data.Input.Count);
        }

        public IActivationFunction Clone()
        {
            return new ArcTanActivationFunction(_f,_df,_fData,_dfData);
        }
    }
}