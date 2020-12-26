using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Statistics;
using NNLib.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace NNLib
{
    public abstract class NormalizationBase
    {
        public async Task FitAndTransform(SupervisedTrainingData data)
        {
            await FitAndTransform(data.TrainingSet.Input);
            if (data.ValidationSet != null)
            {
                await Transform(data.ValidationSet.Input);
            }
            if (data.TestSet != null)
            {
                await Transform(data.TestSet.Input);
            }
        }
        public abstract Task FitAndTransform(IVectorSet set);
        public abstract Task Transform(IVectorSet set);
        public abstract Matrix<double> Transform(Matrix<double> mat);
    }

    public class MinMaxNormalization : NormalizationBase
    {
        private readonly double a;
        private readonly double b;
        private double[] _min;
        private double[] _max;

        public MinMaxNormalization(double a, double b)
        {
            this.a = a;
            this.b = b;
        }

        void ToMinMax(Matrix<double> y)
        {
            for (int i = 0; i < y.RowCount; i++)
            {
                y[i, 0] = (y[i, 0] - _min[i]) / (_max[i] - _min[i] != 0 ? _max[i] - _min[i] : 1);
            }
        }

        public override Task FitAndTransform(IVectorSet set)
        {
            return Task.Run(() =>
            {
                _min = new double[set[0].RowCount];
                _max = new double[set[0].RowCount];
                for (int i = 0; i < set[0].RowCount; i++)
                {
                    double min, max;
                    min = max = set[0][i, 0];
                    for (int j = 0; j < set.Count; j++)
                    {
                        if (set[j][i, 0] < min)
                        {
                            min = set[j].At(i, 0);
                        }

                        if (set[j][i, 0] > max)
                        {
                            max = set[j].At(i, 0);
                        }
                    }

                    _min[i] = min;
                    _max[i] = max;

                    for (int j = 0; j < set.Count; j++)
                    {
                        set[j].At(i, 0, (set[j].At(i, 0) - min) * (b - a) / (max - min != 0 ? max - min : 1) + a);
                    }
                }
            });
        }

        public override Task Transform(IVectorSet set)
        {
            return Task.Run(() =>
            {
                for (int i = 0; i < set.Count; i++)
                {
                    ToMinMax(set[i]);
                }
            });
        }

        public override Matrix<double> Transform(Matrix<double> mat)
        {
            var y = mat.Clone();
            ToMinMax(y);
            return y;
        }
    }

    public class MeanNormalization : NormalizationBase
    {
        private double[] _avg;
        private double[] _max;
        private double[] _min;

        private void ToMean(Matrix<double> y)
        {
            for (int i = 0; i < y.RowCount; i++)
            {
                y[i, 0] = (y[i, 0] - _avg[i]) / (_max[i] - _min[i] != 0 ? _max[i] - _min[i] : 1);
            }
        }

        public override Task FitAndTransform(IVectorSet vec)
        {
            return Task.Run(() =>
            {
                _min = new double[vec[0].RowCount];
                _max = new double[vec[0].RowCount];
                _avg = new double[vec[0].RowCount];
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

                    _min[i] = min;                      
                    _max[i] = max;
                    _avg[i] = avg;

                    for (int j = 0; j < vec.Count; j++)
                    {
                        vec[j][i, 0] = (vec[j][i, 0] - avg) / (max - min != 0 ? max - min : 1);
                    }
                }
            });
        }

        public override Task Transform(IVectorSet set)
        {
            return Task.Run(() =>
            {
                for (int i = 0; i < set.Count; i++)
                {
                    ToMean(set[i]);
                }
            });
        }

        public override Matrix<double> Transform(Matrix<double> mat)
        {
            var y = mat.Clone();
            ToMean(y);
            return y;
        }
    }

    public class Standarization : NormalizationBase
    {
        private double[] _stddev;
        private double[] _avg;

        private void ToStd(Matrix<double> y)
        {
            for (int i = 0; i < y.RowCount; i++)
            {
                y[i, 0] = (y[i, 0] - _avg[i]) / (_stddev[i] == 0d ? 1 : _stddev[i]);
            }
        }

        public override Task FitAndTransform(IVectorSet vec)
        {
            return Task.Run(() =>
            {
                _avg = new double[vec[0].RowCount];
                _stddev = new double[vec[0].RowCount];
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

                    _avg[i] = avg;
                    _stddev[i] = stddev;

                    for (int j = 0; j < vec.Count; j++)
                    {
                        vec[j][i, 0] = (vec[j][i, 0] - avg) / (stddev == 0d ? 1 : stddev);
                    }
                }
            });
        }

        public override Task Transform(IVectorSet set)
        {
            return Task.Run(() =>
            {
                for (int i = 0; i < set.Count; i++)
                {
                    ToStd(set[i]);
                }
            });
        }

        public override Matrix<double> Transform(Matrix<double> mat)
        {
            var y = mat.Clone();
            ToStd(y);
            return y;
        }
    }

    public class RobutstNormalization : NormalizationBase
    {
        private double[] _median;
        private double[] _percDiff;

        private static IEnumerable<Matrix<double>> Enumerate(IEnumerator<Matrix<double>> mat)
        {
            while (mat.MoveNext())
            {
                yield return mat.Current;
            }
        }

        private void ToRobust(Matrix<double> y)
        {
            for (int i = 0; i < y.RowCount; i++)
            {
                var val = (y.At(i, 0) - _median[i]) / _percDiff[i];
                y.At(i, 0, val);
            }
        }

        public override Task FitAndTransform(IVectorSet vec)
        {
            return Task.Run(() =>
            {
                _median = new double[vec[0].RowCount];
                _percDiff = new double[vec[0].RowCount];
                for (int i = 0; i < vec[0].RowCount; i++)
                {
                    var median = Enumerate(vec.GetEnumerator()).Select(m => m.At(i, 0)).Median();
                    var p75 = Enumerate(vec.GetEnumerator()).Select(m => m.At(i, 0)).Percentile(75);
                    var p25 = Enumerate(vec.GetEnumerator()).Select(m => m.At(i, 0)).Percentile(25);
                    var percDiff = p75 - p25 != 0 ? p75 - p25 : 1;

                    _median[i] = median;
                    _percDiff[i] = percDiff;

                    for (int j = 0; j < vec.Count; j++)
                    {
                        var val = (vec[j].At(i, 0) - median) / percDiff;
                        vec[j].At(i, 0, val);
                    }
                }
            });
        }

        public override Task Transform(IVectorSet set)
        {
            return Task.Run(() =>
            {
                for (int i = 0; i < set.Count; i++)
                {
                    ToRobust(set[i]);
                }
            });
        }

        public override Matrix<double> Transform(Matrix<double> mat)
        {
            var y = mat.Clone();
            ToRobust(y);
            return y;
        }
    }
}
