using NNLib.Common;

namespace NNLib.Csv
{
    internal static class CsvVectorSetFactory
    {
        public static (IVectorSet input, IVectorSet target) CreateCsvSupervisedVectorSet(string fileName, SupervisedSetVariableIndexes supervisedSetVariableIndexes, DataSetInfo dataSetInfo)
        {
            var csvDataSetReader = new CsvFileDataSetReader(new CsvInternalReader(fileName), supervisedSetVariableIndexes, dataSetInfo);
            var inputVectorSet = new CsvFileVectorSet(csvDataSetReader, dataSetInfo.SetSize, false);
            var targetVectorSet = new CsvFileVectorSet(csvDataSetReader, dataSetInfo.SetSize, true);

            return (inputVectorSet, targetVectorSet);
        }
    }
}