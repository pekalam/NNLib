using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Statistics;
using NNLib.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace NNLib
{
    public static class Normalization
    {
        private static IEnumerable<Matrix<double>> Enumerate(this IEnumerator<Matrix<double>> mat)
        {
            while (mat.MoveNext())
            {
                yield return mat.Current;
            }
        }

        public static async Task MinMax(SupervisedTrainingData sets, double a = 0, double b = 1)
        {
            void MinMaxVec(IVectorSet vec)
            {
                for (int i = 0; i < vec[0].RowCount; i++)
                {
                    double min, max;
                    min = max = vec[0][i, 0];
                    for (int j = 0; j < vec.Count; j++)
                    {
                        if (vec[j][i, 0] < min)
                        {
                            min = vec[j].At(i, 0);
                        }
                        if (vec[j][i, 0] > max)
                        {
                            max = vec[j].At(i, 0);
                        }
                    }


                    for (int j = 0; j < vec.Count; j++)
                    {
                        vec[j].At(i, 0, (vec[j].At(i, 0) - min) * (b - a) / (max - min != 0 ? max - min : 1) + a);
                    }
                }
            }

            await Task.Run(() =>
            {
                MinMaxVec(sets.ConcatenatedInput);
            });
        }

        public static async Task Mean(SupervisedTrainingData sets)
        {
            void MeanVec(IVectorSet vec)
            {
                for (int i = 0; i < vec[0].RowCount; i++)
                {
                    double min, max, avg = 0;
                    min = max = vec[0][i, 0];
                    for (int j = 0; j < vec.Count; j++)
                    {
                        if (vec[j][i, 0] < min)
                        {
                            min = vec[j][i, 0];
                        }
                        if (vec[j][i, 0] > max)
                        {
                            max = vec[j][i, 0];
                        }

                        avg += vec[j][i, 0] / vec.Count;
                    }


                    for (int j = 0; j < vec.Count; j++)
                    {
                        vec[j][i, 0] = (vec[j][i, 0] - avg) / (max - min != 0 ? max - min : 1);
                    }
                }
            }

            await Task.Run(() =>
            {
                MeanVec(sets.ConcatenatedInput);
            });
        }

        public static async Task Std(SupervisedTrainingData sets)
        {
            void StdVec(IVectorSet vec)
            {
                for (int i = 0; i < vec[0].RowCount; i++)
                {
                    double min, max, avg = 0, stddev = 0;
                    min = max = vec[0][i, 0];
                    for (int j = 0; j < vec.Count; j++)
                    {
                        if (vec[j][i, 0] < min)
                        {
                            min = vec[j][i, 0];
                        }
                        if (vec[j][i, 0] > max)
                        {
                            max = vec[j][i, 0];
                        }

                        avg += vec[j][i, 0] / vec.Count;
                    }

                    for (int j = 0; j < vec.Count; j++)
                    {
                        stddev += Math.Pow(vec[j][i, 0] - avg, 2.0d) / (vec.Count - 1 != 0 ? vec.Count - 1 : 1);
                    }

                    stddev = Math.Sqrt(stddev);

                    for (int j = 0; j < vec.Count; j++)
                    {
                        vec[j][i, 0] = (vec[j][i, 0] - avg) / (stddev == 0d ? 1 : stddev);
                    }
                }
            }

            await Task.Run(() =>
            {
                StdVec(sets.ConcatenatedInput);
            });
        }

        public static async Task Robust(SupervisedTrainingData sets)
        {
            void RobustVec(IVectorSet vec)
            {
                for (int i = 0; i < vec[0].RowCount; i++)
                {
                    var median = vec.GetEnumerator().Enumerate().Select(m => m.At(i, 0)).Median();
                    var p75 = vec.GetEnumerator().Enumerate().Select(m => m.At(i, 0)).Percentile(75);
                    var p25 = vec.GetEnumerator().Enumerate().Select(m => m.At(i, 0)).Percentile(25);
                    var percDiff = p75 - p25 != 0 ? p75 - p25 : 1;

                    for (int j = 0; j < vec.Count; j++)
                    {
                        var val = (vec[j].At(i, 0) - median) / percDiff;
                        vec[j].At(i, 0, val);
                    }
                }
            }

            await Task.Run(() =>
            {
                RobustVec(sets.ConcatenatedInput);
            });
        }
    }
}
