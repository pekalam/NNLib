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


        private int CalculateRowCountInPage(string fileName, int pageSize)
        {
            var rowSz = _fileAnalyzer.GetRowSizeApproximation(fileName);

            int pageSz = pageSize / rowSz;
            return pageSz;
        }

        private List<FilePart> PartitionDataSet(List<long> fileNewLinePositions, int pageSize, long start)
        {
            var parts = new List<FilePart>();
            int i = 0;
            foreach (var pos in fileNewLinePositions)
            {
                i++;

                if (i == pageSize)
                {
                    parts.Add(new FilePart(start, pos, i));
                    start = pos;
                    i = 0;
                }
            }

            if (i != 0)
            {
                parts.Add(new FilePart(start, fileNewLinePositions[^1], i));
            }

            return parts;
        }

        public DataSetInfo[] CreatePartitionedDataSets(string fileName, DataSetDivisionOptions divisionOpt)
        {
            Log.Logger.Debug("Creating partitioned dataSet from file: {fileName} with options {@options}", fileName,
                divisionOpt);

            var positions = _fileAnalyzer.GetDataItemsNewLinePositions(fileName);
            var rowCountInPage = CalculateRowCountInPage(fileName, divisionOpt.PageSize);
            Log.Logger.Debug("Row count per page: {@count}", rowCountInPage);

            var variableNames = _fileAnalyzer.GetVariableNames(fileName);
            Log.Logger.Debug("File {file} has following variables: {@vars}", fileName, variableNames);

            var divisions = _divider.Divide(positions, divisionOpt);
            Log.Logger.Debug("Divided file into {count} sets", divisions.Length);

            var setInfos = new DataSetInfo[divisions.Length];

            long previousStart = 0;
            for (int i = 0; i < setInfos.Length; i++)
            {
                var fileParts = PartitionDataSet(divisions[i].positions, rowCountInPage, previousStart);
                Log.Logger.Debug("{count} file parts for {type} starting at {start} with {n} dataItems end: {end}", fileParts.Count,
                    divisions[i].setType, fileParts[0].Offset, fileParts[0].DataItems, fileParts[0].End);
                previousStart = divisions[i].positions[^1];
                var setInfo =
                    new DataSetInfo(fileParts, divisions[i].positions.Count, rowCountInPage, divisions[i].setType,
                        variableNames);
                setInfos[i] = setInfo;
            }

            return setInfos;
        }
    }
}