using System;
using MathNet.Numerics.LinearAlgebra;
using NNLib.Data;

namespace NNLib.ActivationFunction
{
    public class SigmoidActivationFunction : IActivationFunction
    {
        private Matrix<double> _f = null!;
        private Matrix<double> _df = null!;
        private Matrix<double> _dfClone = null!;

        private NetDataMatrixPool? _fData;
        private NetDataMatrixPool? _dfData;
        private NetDataMatrixPool? _dfDataClone;

        private SigmoidActivationFunction(Matrix<double> f, Matrix<double> df, Matrix<double> dfClone, NetDataMatrixPool? fData, NetDataMatrixPool? dfData, NetDataMatrixPool? dfDataClone)
        {
            _f = f.Clone();
            _df = df.Clone();
            _dfClone = dfClone.Clone();
            _fData = fData?.Clone();
            _dfData = dfData?.Clone();
            _dfDataClone = dfDataClone?.Clone();
        }

        public SigmoidActivationFunction()
        {
            
        }

        public Matrix<double> Function(Matrix<double> x)
        {
            var cols = x.ColumnCount;
            Matrix<double> storage = cols == _f.ColumnCount ? _f : _fData!.Get(cols);

            x.Negate(storage);
            storage.PointwiseExp(storage);
            storage.Add(1, storage);
            storage.PointwisePower(-1, storage);

            return storage;
        }

        public Matrix<double> Derivative(Matrix<double> x)
        {
            var cols = x.ColumnCount;

            Matrix<double> storage = cols == _df.ColumnCount ? _df : _dfData!.Get(cols);
            Matrix<double> storage2 = cols == _df.ColumnCount ? _dfClone : _dfDataClone!.Get(cols);

            x.Negate(storage);
            storage.PointwiseExp(storage);
            storage.Add(1, storage);
            storage.PointwisePower(-1, storage);

            storage.Negate(storage2);
            storage2.Add(1, storage2);

            storage.PointwiseMultiply(storage2, storage);

            return storage;
        }

        public void InitMemory(Layer layer)
        {
            _f = Matrix<double>.Build.Dense(layer.NeuronsCount, 1);
            _df = Matrix<double>.Build.Dense(layer.NeuronsCount, 1);
            _dfClone = _df.Clone();
        }

        public void InitMemoryForData(Layer layer, SupervisedTrainingSamples data)
        {
            _fData = new NetDataMatrixPool(layer.NeuronsCount, data.Input.Count);
            _dfData = new NetDataMatrixPool(layer.NeuronsCount, data.Input.Count);
            _dfDataClone = new NetDataMatrixPool(layer.NeuronsCount, data.Input.Count);
        }

        public IActivationFunction Clone()
        {
            return new SigmoidActivationFunction(_f, _df,_dfClone,_fData,_dfData,_dfDataClone);
        }
    }
}