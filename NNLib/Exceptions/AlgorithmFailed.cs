using System;
using System.Runtime.Serialization;

namespace NNLib
{
    public class AlgorithmFailed : Exception
    {
        public AlgorithmFailed()
        {
        }

        public AlgorithmFailed(string? message) : base(message)
        {
        }

        public AlgorithmFailed(string? message, Exception? innerException) : base(message, innerException)
        {
        }
    }
}