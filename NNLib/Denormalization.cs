using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Statistics;
using NNLib.Data;

namespace NNLib
{
    public static class Denormalization
    {
        private static (double min, double max) FindMinMax(IVectorSet vectorSet, int row)
        {
            double min = double.MaxValue;
            double max = double.MinValue;

            foreach (var mat in vectorSet)
            {
                if (mat.At(row, 0) < min) min = mat.At(row, 0);
                if (mat.At(row, 0) > max) max = mat.At(row, 0);
            }

            return (min, max);
        }

        private static (double avg, double min, double max) FindMean(IVectorSet vec, int row)
        {
            double avg = 0;
            double min = double.MaxValue;
            double max = double.MinValue;
            foreach (var mat in vec)
            {
                if (mat[row, 0] < min)
                {
                    min = mat[row, 0];
                }
                if (mat[row, 0] > max)
                {
                    max = mat[row, 0];
                }

                avg += mat[row, 0] / vec.Count;
            }

            return (avg, min, max);
        }

        public static Matrix<double> ToMinMax(Matrix<double> inputMat, IVectorSet totalInput)
        {
            var y = inputMat.Clone();

            for (int i = 0; i < inputMat.RowCount; i++)
            {
                var (min, max) = FindMinMax(totalInput, i);

                y[i, 0] = (y[i, 0] - min) / (max - min != 0 ? max - min : 1);
            }

            return y;
        }


        public static Matrix<double> ToMean(Matrix<double> inputMat, IVectorSet totalInput)
        {
            var y = inputMat.Clone();

            for (int i = 0; i < inputMat.RowCount; i++)
            {
                var (avg, min, max) = FindMean(totalInput, i);

                y[i, 0] = (y[i, 0] - avg) / (max - min != 0 ? max - min : 1);
            }

            return y;
        }


        public static Matrix<double> ToStd(Matrix<double> inputMat, IVectorSet totalInput)
        {
            var y = inputMat.Clone();

            double stddev = 0;
            for (int i = 0; i < inputMat.RowCount; i++)
            {
                var (avg, min, max) = FindMean(totalInput, i);

                for (int j = 0; j < totalInput.Count; j++)
                {
                    stddev += Math.Pow(totalInput[j][i, 0] - avg, 2.0d) / (totalInput.Count - 1 != 0 ? totalInput.Count - 1 : 1);
                }
                stddev = Math.Sqrt(stddev);

                y[i, 0] = (y[i, 0] - avg) / (stddev == 0d ? 1 : stddev);
            }

            return y;
        }

        public static Matrix<double> ToRobust(Matrix<double> inputMat, IVectorSet totalInput)
        {
            IEnumerable<Matrix<double>> Enumerate(IEnumerator<Matrix<double>> mat)
            {
                while (mat.MoveNext())
                {
                    yield return mat.Current;
                }
            }

            var y = inputMat.Clone();

            for (int i = 0; i < totalInput[0].RowCount; i++)
            {
                var median = Enumerate(totalInput.GetEnumerator()).Select(m => m.At(i, 0)).Median();
                var p75 = Enumerate(totalInput.GetEnumerator()).Select(m => m.At(i, 0)).Percentile(75);
                var p25 = Enumerate(totalInput.GetEnumerator()).Select(m => m.At(i, 0)).Percentile(25);
                var percDiff = p75 - p25 != 0 ? p75 - p25 : 1;

                var val = (y.At(i, 0) - median) / percDiff;
                y.At(i, 0, val);
            }

            return y;
        }
    }
}