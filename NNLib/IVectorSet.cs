using System;
using MathNet.Numerics.LinearAlgebra;

namespace NNLib
{
    public interface IVectorSet : IDisposable
    {
        Matrix<double> this[int index] { get; set; }
        int Count { get; }
    }
}