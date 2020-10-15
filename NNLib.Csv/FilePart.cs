using System;

namespace NNLib.Csv
{
    internal class FilePart
    {
        public FilePart(long offset, long end, int dataItems)
        {
            if (dataItems == 0)
            {
                throw new ArgumentException("DataItems cannot be 0");
            }

            if (end <= offset)
            {
                throw new ArgumentException($"Offset ({offset}) cannot be greater or equal end ({end})");
            }

            if (end <= 0)
            {
                throw new ArgumentException($"Invalid end value {end}");
            }

            if (offset < 0)
            {
                throw new ArgumentException("Offset is less than 0");
            }

            Offset = offset;
            End = end;
            DataItems = dataItems;
        }

        public readonly long Offset;
        public readonly long End;
        public readonly int DataItems;
    }
}