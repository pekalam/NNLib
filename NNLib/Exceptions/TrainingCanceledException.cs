using System;
using System.Runtime.CompilerServices;
using System.Threading;

namespace NNLib.Exceptions
{
    public class TrainingCanceledException : Exception
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ThrowIfCancellationRequested(in CancellationToken ct)
        {
            if (ct.IsCancellationRequested)
            {
                throw new TrainingCanceledException();
            }
        }
    }
}