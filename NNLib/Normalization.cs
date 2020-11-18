using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Statistics;
using NNLib.Data;

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

        public static async Task MinMax(SupervisedTrainingData sets)
        {
            void MinMax(SupervisedTrainingSamples set)
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
                            if (max == min)
                            {
                                vec[j].At(i, 0, 0);
                            }
                            else
                            {
                                vec[j].At(i, 0, (vec[j].At(i, 0) - min) / (max - min));
                            }
                        }
                    }
                }
                MinMaxVec(set.Input);
            }

            await Task.Run(() =>
            {
                MinMax(sets.TrainingSet);
                if (sets.ValidationSet != null) MinMax(sets.ValidationSet);
                if (sets.TestSet != null) MinMax(sets.TestSet);

            });
        }

        public static async Task Mean(SupervisedTrainingData sets)
        {
            void Mean(SupervisedTrainingSamples set)
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
                            if (max == min)
                            {
                                vec[j][i, 0] = 0;
                            }
                            else
                            {
                                vec[j][i, 0] = (vec[j][i, 0] - avg) / (max - min);
                            }
                        }
                    }
                }
                MeanVec(set.Input);
            }


            await Task.Run(() =>
            {
                Mean(sets.TrainingSet);
                if (sets.ValidationSet != null) Mean(sets.ValidationSet);
                if (sets.TestSet != null) Mean(sets.TestSet);
            });
        }

        public static async Task Std(SupervisedTrainingData sets)
        {
            void Std(SupervisedTrainingSamples set)
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
                            stddev += Math.Pow(vec[j][i, 0] - avg, 2.0d) / (vec.Count - 1);
                        }

                        stddev = Math.Sqrt(stddev);

                        for (int j = 0; j < vec.Count; j++)
                        {
                            vec[j][i, 0] = (vec[j][i, 0] - avg) / (stddev == 0d ? 1 : stddev);
                        }
                    }
                }
                StdVec(set.Input);
            }

            await Task.Run(() =>
            {
                Std(sets.TrainingSet);
                if (sets.ValidationSet != null) Std(sets.ValidationSet);
                if (sets.TestSet != null) Std(sets.TestSet);
            });
        }

        public static async Task Robust(SupervisedTrainingData sets)
        {
            void Robust(SupervisedTrainingSamples set)
            {
                void RobustVec(IVectorSet vec)
                {
                    for (int i = 0; i < vec[0].RowCount; i++)
                    {
                        var median = vec.GetEnumerator().Enumerate().Select(m => m.At(i, 0)).Median();
                        var p75 = vec.GetEnumerator().Enumerate().Select(m => m.At(i, 0)).Percentile(75);
                        var p25 = vec.GetEnumerator().Enumerate().Select(m => m.At(i, 0)).Percentile(25);

                        for (int j = 0; j < vec.Count; j++)
                        {
                            var val = (vec[j].At(i, 0) - median) / (p75 - p25);
                            vec[j].At(i, 0, val);
                        }
                    }
                }
                RobustVec(set.Input);
            }

            await Task.Run(() =>
            {
                Robust(sets.TrainingSet);
                if (sets.ValidationSet != null) Robust(sets.ValidationSet);
                if (sets.TestSet != null) Robust(sets.TestSet);
            });
        }
    }
}
