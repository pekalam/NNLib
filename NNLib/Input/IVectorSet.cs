using MathNet.Numerics.LinearAlgebra;
using System;

namespace NNLib
{
    public interface IVectorSet : IDisposable
    {
        Matrix<double> this[int index] { get; set; }
        int Count { get; }
    }
}