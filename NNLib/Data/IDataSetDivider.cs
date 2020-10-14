using System.Collections.Generic;

namespace NNLib.Common
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