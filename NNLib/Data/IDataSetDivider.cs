using System.Collections.Generic;

namespace NNLib.Common
{
    public interface IDataSetDivider
    {
        (DataSetType setType, List<long> positions)[] Divide(List<long> fileNewLinePositions,
            DataSetDivisionOptions divOptions);
    }
}