﻿using NNLib.Data;
using System;
using System.Diagnostics;
using System.Linq;

namespace NNLib.Csv
{
    internal class DataSetInfoBuilder
    {
        private readonly CsvFileAnalyzer _fileAnalyzer= new CsvFileAnalyzer();
        private readonly IDataSetDivider _divider;

        public DataSetInfoBuilder(IDataSetDivider divider)
        {
            _divider = divider;
        }

        public DataSetInfo[] CreatePartitionedDataSets(string fileName, DataSetDivisionOptions divisionOpt)
        {
            Debug.WriteLine("Creating partitioned dataSet from file: {0} with options {1}", fileName,
                divisionOpt);

            var positions = _fileAnalyzer.GetDataItemsNewLinePositions(fileName);
            Debug.WriteLine("Row count: {0}", positions.Count);

            var variableNames = _fileAnalyzer.GetVariableNames(fileName);
            Debug.WriteLine("File {0} has following variables: {1}", fileName, variableNames);

            var divisions = _divider.Divide(positions, divisionOpt);
            Debug.WriteLine("Divided file into {0} sets", divisions.Length);

            var setInfos = new DataSetInfo[divisions.Length];

            var orgDivisionsOrder = divisions.SelectMany(d => d.positions).OrderBy(p => p).ToArray();

            bool isLinear = true;
            int di = 0;
            foreach (var d in divisions.SelectMany(v => v.positions))
            {
                if (orgDivisionsOrder[di++] != d)
                {
                    isLinear = false;
                    break;
                }
            }

            if (!isLinear)
            {
                for (int i = 0; i < setInfos.Length; i++)
                {
                    var fileParts = new FilePart[divisions[i].positions.Count];
                    for (int j = 0; j < fileParts.Length; j++)
                    {
                        var ind = Array.BinarySearch(orgDivisionsOrder, divisions[i].positions[j]);

                        long previousStart = 0;
                        if (ind == 0)
                        {
                            previousStart = 0;
                        }
                        else
                        {
                            previousStart = orgDivisionsOrder[ind - 1];
                        }

                        var filePart = new FilePart(previousStart, divisions[i].positions[j], 1);
                        // Debug.WriteLine("file part for {0} starting at {1} with {2} dataItems end: {3}",
                        //     divisions[i].setType, filePart.Offset, filePart.DataItems, filePart.End);
                        fileParts[j] = filePart;
                    }

                    var setInfo =
                        new DataSetInfo(fileParts, divisions[i].positions.Count, -0, divisions[i].setType,
                            variableNames);
                    setInfos[i] = setInfo;
                }
            }else
            {
                long previousStart = 0;
                for (int i = 0; i < setInfos.Length; i++)
                {
                    var filePart = new FilePart(previousStart, divisions[i].positions[^1], divisions[i].positions.Count);
                    Debug.WriteLine("file part for {0} starting at {1} with {2} dataItems end: {3}",
                        divisions[i].setType, filePart.Offset, filePart.DataItems, filePart.End);
                    previousStart = divisions[i].positions[^1];
                    var setInfo =
                        new DataSetInfo(new []{filePart}, divisions[i].positions.Count, -0, divisions[i].setType,
                            variableNames);
                    setInfos[i] = setInfo;
                }
            }



            return setInfos;
        }
    }
}