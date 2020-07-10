using System.Collections.Generic;
using NNLib.Common;
using Serilog;

namespace NNLib.Csv
{
    internal class InternalDataSetBuilder
    {
        private readonly CsvDataSetFileAnalyzer _fileAnalyzer;
        private readonly IDataSetDivider _divider;

        public InternalDataSetBuilder(IDataSetDivider divider)
        {
            _fileAnalyzer = new CsvDataSetFileAnalyzer();
            _divider = divider;
        }

        public DataSetInfo[] CreatePartitionedDataSets(string fileName, DataSetDivisionOptions divisionOpt)
        {
            Log.Logger.Debug("Creating partitioned dataSet from file: {fileName} with options {@options}", fileName,
                divisionOpt);

            var positions = _fileAnalyzer.GetDataItemsNewLinePositions(fileName);
            Log.Logger.Debug("Row count: {@count}", positions.Count);

            var variableNames = _fileAnalyzer.GetVariableNames(fileName);
            Log.Logger.Debug("File {file} has following variables: {@vars}", fileName, variableNames);

            var divisions = _divider.Divide(positions, divisionOpt);
            Log.Logger.Debug("Divided file into {count} sets", divisions.Length);

            var setInfos = new DataSetInfo[divisions.Length];

            long previousStart = 0;
            for (int i = 0; i < setInfos.Length; i++)
            {
                var filePart = new FilePart(previousStart, divisions[i].positions[^1], divisions[i].positions.Count);
                Log.Logger.Debug("file part for {type} starting at {start} with {n} dataItems end: {end}",
                    divisions[i].setType, filePart.Offset, filePart.DataItems, filePart.End);
                previousStart = divisions[i].positions[^1];
                var setInfo =
                    new DataSetInfo(filePart, divisions[i].positions.Count, -0, divisions[i].setType,
                        variableNames);
                setInfos[i] = setInfo;
            }

            return setInfos;
        }
    }
}