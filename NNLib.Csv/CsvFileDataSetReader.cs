using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using MathNet.Numerics.LinearAlgebra;
using NNLib.Common;

[assembly: InternalsVisibleTo("Data.Tests")]
[assembly: InternalsVisibleTo("Infrastructure.Data.Tests")]
namespace NNLib.Csv
{
    internal class CsvFileDataSetReader : IDisposable
    {
        private readonly DataSetInfo _dataSetInfo;
        private readonly List<(Matrix<double> input, Matrix<double> target)> _read = new List<(Matrix<double> input, Matrix<double> target)>();
        private int _divisionIndex;
        private (int start, int end) _readRange;
        private bool _needsReload;

        private SupervisedSetVariableIndexes _setVariableIndexes;
        private readonly ICsvInternalReader _csvInternalReader;

        public CsvFileDataSetReader(ICsvInternalReader csvInternalReader, SupervisedSetVariableIndexes setVariableIndexes, DataSetInfo dataDataSetInfo)
        {
            _setVariableIndexes = setVariableIndexes;
            _dataSetInfo = dataDataSetInfo;
            _csvInternalReader = csvInternalReader;
        }

        public (int start, int end) ReadRange => _readRange;

        private void ReadNextParts(int vectorIndex)
        {
            _read.Clear();
            int i = 0;
            int items = 0;
            int readStart = 0;
            while (i < _dataSetInfo.FileParts.Count)
            {
                items += _dataSetInfo.FileParts[i].DataItems;
                if (items > vectorIndex)
                {
                    _divisionIndex = i;
                    break;
                }

                readStart += _dataSetInfo.FileParts[i].DataItems;
                i++;
            }

            if (i == _dataSetInfo.FileParts.Count)
            {
                throw new ArgumentOutOfRangeException($"vectorIndex {vectorIndex} out of range");
            }
            var filePart = _dataSetInfo.FileParts[_divisionIndex];
            var vectorSets = _csvInternalReader.ReadVectorSets(filePart, _setVariableIndexes);

            _readRange = (readStart, readStart + filePart.DataItems);

            _read.AddRange(vectorSets);
        }

        internal void ChangeVariables(SupervisedSetVariableIndexes newVariableIndexes)
        {
            _setVariableIndexes = newVariableIndexes;
            _needsReload = true;
        }

        internal Matrix<double> ReadAt(bool targetSet, int vectorIndex)
        {
            lock (_read)
            {
                if (vectorIndex < _readRange.start || vectorIndex >= _readRange.end || _needsReload || _read.Count == 0)
                {
                    ReadNextParts(vectorIndex);
                    _needsReload = false;
                }

                Matrix<double> vectorSet;
                if (targetSet)
                {
                    vectorSet = _read[vectorIndex % _read.Count].target;
                }
                else
                {
                    vectorSet = _read[vectorIndex % _read.Count].input;
                }

                return vectorSet;
            }
        }

        public void Dispose()
        {
            _csvInternalReader.Dispose();
            _read.Clear();
        }
    }
}
