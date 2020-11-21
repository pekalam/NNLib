using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using System;

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


    /// <summary>
    /// Sets values of weights using values from normal distribution with standard deviation equal 0.01. Biases are set to 0.
    /// </summary>
    public class SmallStdevNormDistMatrixBuilder : NormDistMatrixBuilder
    {
        public SmallStdevNormDistMatrixBuilder() : base(new NormDistMatrixBuilderOptions()
        {
            BMean = 0,BStdDev = 0, WMean =0, WStdDev = 0.01
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

        public NormDistMatrixBuilderOptions Clone()
        {
            return new NormDistMatrixBuilderOptions
            {
                BMean = BMean,BStdDev = BStdDev,WMean = WMean,WStdDev = WStdDev,
            };
        }
    }

    /// <summary>
    /// Sets values of weights and biases using values from normal distribution.
    /// </summary>
    public class NormDistMatrixBuilder : MatrixBuilder
    {
        public NormDistMatrixBuilder(NormDistMatrixBuilderOptions options)
        {
            Options = options;
        }

        public NormDistMatrixBuilderOptions Options { get; }

        public override void BuildAllMatrices(int neuronsCount, int inputsCount, Layer layer)
        {
            layer.Weights = Matrix<double>.Build.Random(neuronsCount, inputsCount, new Normal(Options.WMean, Options.WStdDev));
            layer.Biases = Matrix<double>.Build.Random(neuronsCount, 1, new Normal(Options.BMean, Options.BStdDev));
        }

        public override void AdjustMatrices(int neuronsCount, int inputsCount, Layer layer)
        {
            if (inputsCount > layer.InputsCount)
            {
                while (inputsCount != layer.InputsCount)
                {
                    var col = Matrix<double>.Build.Random(layer.NeuronsCount, 1, new Normal(Options.WMean, Options.WStdDev)).Column(0);
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
                    var Wrow = Matrix<double>.Build.Random(1, layer.InputsCount, new Normal(Options.WMean, Options.WStdDev)).Row(0);
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

    /// <summary>
    /// Sets weights with commonly used heuristics: U[-1/sqrt(inputsCount),1/sqrt(inputsCount)]. Biases are set to 0.
    /// </summary>
    public class SmallNumbersMatrixBuilder : MatrixBuilder
    {
        public override void BuildAllMatrices(int neuronsCount, int inputsCount, Layer layer)
        {
            var a = Math.Sqrt(1d / inputsCount);
            layer.Weights = Matrix<double>.Build.Random(neuronsCount, inputsCount, new ContinuousUniform(-a, a));
            layer.Biases = Matrix<double>.Build.Dense(neuronsCount, 1, 0);
        }

        public override void AdjustMatrices(int neuronsCount, int inputsCount, Layer layer)
        {
            var a = Math.Sqrt(1d / inputsCount);
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

    /// <summary>
    /// Implements Xavier weights initialization method.
    /// </summary>
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

    /// <summary>
    /// Implements Nguyen-Widrow weights initialization method.
    /// </summary>
    public class NguyenWidrowMatrixBuilder : MatrixBuilder
    {
        private void CheckCanInit(Layer layer)
        {
            if (!(layer.IsOutputLayer || layer.Network!.BaseLayers[0] == layer))
            {
                throw new ArgumentException("Layer must be an output layer or first hidden");
            }
        }

        public override void BuildAllMatrices(int neuronsCount, int inputsCount, Layer layer)
        {
            CheckCanInit(layer);

            if (!layer.IsOutputLayer)
            {
                double h = 0.7 * Math.Pow(neuronsCount, 1d / inputsCount);
                var initialW = layer.Weights = Matrix<double>.Build.Random(neuronsCount, inputsCount, new ContinuousUniform(-.5d, .5d));
                layer.Biases = Matrix<double>.Build.Random(neuronsCount, 1, new ContinuousUniform(-h, h));

                //for each neuron
                for (int i = 0; i < initialW.RowCount; i++)
                {
                    double wRoot = 0d;
                    for (int j = 0; j < initialW.ColumnCount; j++)
                    {
                        wRoot += initialW.At(i, j) * initialW.At(i, j);
                    }
                    wRoot = Math.Sqrt(wRoot);

                    for (int j = 0; j < initialW.ColumnCount; j++)
                    {
                        layer.Weights.At(i, j, initialW.At(i, j) * h / wRoot);
                    }
                }
            }
            else
            {
                layer.Weights = Matrix<double>.Build.Random(neuronsCount, inputsCount, new ContinuousUniform(-.5d, .5d));
                layer.Biases = Matrix<double>.Build.Dense(neuronsCount, 1, 0);
            }
        }

        public override void AdjustMatrices(int neuronsCount, int inputsCount, Layer layer)
        {
            BuildAllMatrices(neuronsCount, inputsCount, layer);
        }
    }

    /// <summary>
    /// Sets weights using values from uniform distribution with standard deviation equal sqrt(inputsCount).
    /// </summary>
    public class SqrMUniformMatrixBuilder : MatrixBuilder
    {
        public override void BuildAllMatrices(int neuronsCount, int inputsCount, Layer layer)
        {
            var a = Math.Sqrt(12d) / (Math.Sqrt(inputsCount) * 2d);
            layer.Weights = Matrix<double>.Build.Random(neuronsCount, inputsCount, new ContinuousUniform(-a, a));
            layer.Biases = Matrix<double>.Build.Dense(neuronsCount, 1, 0);
        }

        public override void AdjustMatrices(int neuronsCount, int inputsCount, Layer layer)
        {
            BuildAllMatrices(neuronsCount, inputsCount, layer);
        }
    }
}