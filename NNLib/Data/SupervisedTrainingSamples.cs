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

        public Matrix<double> ReadInputSamples()
        {
            Matrix<double> I = Matrix<double>.Build.Dense(Input[0].RowCount, Input.Count);
            var iCount = Input.Count;
            for (int i = 0; i < iCount; i++)
            {
                I.SetColumn(i, Input[i].AsColumnMajorArray());
            }

            return I;
        }

        public Matrix<double> ReadTargetSamples()
        {
            Matrix<double> T = Matrix<double>.Build.Dense(Target[0].RowCount, Target.Count);
            var tCount = Target.Count;
            for (int i = 0; i < tCount; i++)
            {
                T.SetColumn(i, Target[i].AsColumnMajorArray());
            }

            return T;
        }



        public void Dispose()
        {
            Input?.Dispose();
            Target?.Dispose();
        }
    }
}