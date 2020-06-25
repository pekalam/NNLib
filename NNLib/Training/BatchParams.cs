using System;

namespace NNLib.Training
{
    //todo remove
    public class BatchParams
    {
        private int _batchSize = 1;

        public int BatchSize
        {
            get => _batchSize;
            set
            {
                if (value <= 0)
                {
                    throw new InvalidOperationException();
                }
                _batchSize = value;
            }
        }
    }
}