using System;
using System.Runtime.Serialization;

namespace NNLib
{
    public class AlgorithmFailed : Exception
    {
        public AlgorithmFailed()
        {
        }

        protected AlgorithmFailed(SerializationInfo? info, StreamingContext context) : base(info, context)
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