using System;

namespace NNLib.Exceptions
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