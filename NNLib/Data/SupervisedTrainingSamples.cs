using System;
using MathNet.Numerics.LinearAlgebra;

namespace NNLib.Data
{
    /// <summary>
    /// Contains input and target vector sets. Used by supervised training algorithms.
    /// </summary>
    public partial class SupervisedTrainingSamples : IDisposable
    {
        public IVectorSet Input { get; }
        public IVectorSet Target { get; }

        public SupervisedTrainingSamples(IVectorSet input, IVectorSet target)
        {
            if (input.Count != target.Count)
            {
                throw new ArgumentException($"Invalid count of input and target sets: {input.Count} {target.Count}");
            }

            Input = input;
            Target = target;
        }

        public (Matrix<double> input, Matrix<double> target) ReadAllSamples()
        {
            Matrix<double> I = Matrix<double>.Build.Dense(Input[0].RowCount, Input.Count);
            Matrix<double> T = Matrix<double>.Build.Dense(Target[0].RowCount, Target.Count);

            for (int i = 0; i < Input.Count; i++)
            {
                I.SetColumn(i, Input[i].AsColumnMajorArray());
                T.SetColumn(i, Target[i].AsColumnMajorArray());
            }

            return (I, T);
        }

        public void Dispose()
        {
            Input?.Dispose();
            Target?.Dispose();
        }
    }
}