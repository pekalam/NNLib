using System;
using System.Diagnostics;
using System.Linq;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;

namespace NNLib.MLP
{
    /// <summary>
    /// Base class for matrix builders which contain strategy for layer parameters initializaion.
    /// </summary>
    public abstract class MatrixBuilder
    {
        public abstract void BuildAllMatrices(int neuronsCount, int inputsCount, Layer layer);
        /// <summary>
        /// Adjust layer parameters if <see cref="neuronsCount"/> or <see cref="inputsCount"/> of network are changed.
        /// </summary>
        public abstract void AdjustMatrices(int neuronsCount, int inputsCount, Layer layer);
    }


    public class DefaultNormDistMatrixBuilder : NormDistMatrixBuilder
    {
        public DefaultNormDistMatrixBuilder() : base(new NormDistMatrixBuilderOptions()
        {
            BMean = 0,BStdDev = 0, WMean =0, WStdDev = 1
        })
        {
        }
    }


    public class NormDistMatrixBuilderOptions
    {
        public double WStdDev { get; set; }
        public double WMean { get; set; }
        public double BStdDev { get; set; }
        public double BMean { get; set; }
    }

    /// <summary>
    /// Sets values of weight and biases matrices as values from normal distribution
    /// </summary>
    public class NormDistMatrixBuilder : MatrixBuilder
    {
        private readonly NormDistMatrixBuilderOptions _options;

        public NormDistMatrixBuilder(NormDistMatrixBuilderOptions options)
        {
            _options = options;
        }

        public override void BuildAllMatrices(int neuronsCount, int inputsCount, Layer layer)
        {
            layer.Weights = Matrix<double>.Build.Random(neuronsCount, inputsCount, new Normal(_options.WMean, _options.WStdDev));
            layer.Biases = Matrix<double>.Build.Random(neuronsCount, 1, new Normal(_options.BMean, _options.BStdDev));
        }

        public override void AdjustMatrices(int neuronsCount, int inputsCount, Layer layer)
        {
            if (inputsCount > layer.InputsCount)
            {
                while (inputsCount != layer.InputsCount)
                {
                    var col = Matrix<double>.Build.Random(layer.NeuronsCount, 1, new Normal(_options.WMean, _options.WStdDev)).Column(0);
                    layer.Weights = layer.Weights.InsertColumn(layer.Weights.ColumnCount, col);
                }
            }
            else if (inputsCount < layer.InputsCount)
            {
                while (inputsCount != layer.InputsCount)
                {
                    layer.Weights = layer.Weights.RemoveColumn(layer.Weights.ColumnCount - 1);
                }
            }

            if (neuronsCount > layer.NeuronsCount)
            {
                while (neuronsCount != layer.NeuronsCount)
                {
                    var Wrow = Matrix<double>.Build.Random(1, layer.InputsCount, new Normal(_options.WMean, _options.WStdDev)).Row(0);
                    var Brow = Matrix<double>.Build.Dense(1, 1, 0).Column(0);
                    layer.Weights = layer.Weights.InsertRow(layer.Weights.RowCount, Wrow);
                    layer.Biases = layer.Biases.InsertRow(layer.Biases.RowCount, Brow);
                }
            }
            else if (neuronsCount < layer.NeuronsCount)
            {
                while (neuronsCount != layer.NeuronsCount)
                {
                    layer.Weights = layer.Weights.RemoveRow(layer.Weights.RowCount - 1);
                    layer.Biases = layer.Biases.RemoveRow(layer.Biases.RowCount - 1);
                }
            }
        }
    }


    public class XavierMatrixBuilder : MatrixBuilder
    {
        public override void BuildAllMatrices(int neuronsCount, int inputsCount, Layer layer)
        {
            var a = Math.Sqrt(6d / (inputsCount + neuronsCount));
            layer.Weights = Matrix<double>.Build.Random(neuronsCount, inputsCount, new ContinuousUniform(-a, a));
            layer.Biases = Matrix<double>.Build.Dense(neuronsCount, 1, 0);
        }

        public override void AdjustMatrices(int neuronsCount, int inputsCount, Layer layer)
        {
            var a = Math.Sqrt(6d / (inputsCount + neuronsCount));

            if (inputsCount != layer.InputsCount)
            {
                layer.Weights = Matrix<double>.Build.Random(neuronsCount, inputsCount, new ContinuousUniform(-a, a));
            }

            if (neuronsCount != layer.NeuronsCount)
            {
                layer.Weights = Matrix<double>.Build.Random(neuronsCount, inputsCount, new ContinuousUniform(-a, a));
                layer.Biases = Matrix<double>.Build.Dense(neuronsCount, 1, 0);
            }
        }
    }
}