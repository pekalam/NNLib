using NNLib.Data;
using System.Collections.Generic;

namespace NNLib.Csv
{
    /// <summary>
    /// Service responsible for dividing training data.
    /// </summary>
    public interface IDataSetDivider
    {
        (DataSetType setType, List<long> positions)[] Divide(List<long> fileNewLinePositions,
            DataSetDivisionOptions divOptions);
    }
}