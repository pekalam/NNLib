﻿using System;
using System.Linq;
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
        private (Matrix<double> input, Matrix<double> target)[] _fileContents;

        private SupervisedSetVariableIndexes _setVariableIndexes;
        private readonly ICsvInternalReader _csvInternalReader;

        private CsvFileDataSetReader(SupervisedSetVariableIndexes setVariableIndexes, DataSetInfo dataSetInfo, (Matrix<double> input, Matrix<double> target)[] fileContents, ICsvInternalReader csvInternalReader)
        {
            _dataSetInfo = dataSetInfo;
            _setVariableIndexes = setVariableIndexes;
            _fileContents = fileContents;
            _csvInternalReader = csvInternalReader;
        }

        public CsvFileDataSetReader(ICsvInternalReader csvInternalReader,
            SupervisedSetVariableIndexes setVariableIndexes, DataSetInfo dataDataSetInfo)
        {
            _setVariableIndexes = setVariableIndexes;
            _dataSetInfo = dataDataSetInfo;
            _csvInternalReader = csvInternalReader;
            _fileContents = csvInternalReader.ReadVectorSets(dataDataSetInfo.FilePart, setVariableIndexes);
        }

        internal void ChangeVariables(SupervisedSetVariableIndexes newVariableIndexes)
        {
            _setVariableIndexes = newVariableIndexes;
            _fileContents = _csvInternalReader.ReadVectorSets(_dataSetInfo.FilePart, _setVariableIndexes);
        }

        internal Matrix<double> ReadAt(bool targetSet, int vectorIndex)
        {
            Matrix<double> vectorSet;
            if (targetSet)
            {
                vectorSet = _fileContents[vectorIndex].target;
            }
            else
            {
                vectorSet = _fileContents[vectorIndex].input;
            }

            return vectorSet;
        }

        internal void WriteAt(bool targetSet, int vectorIndex, Matrix<double> mat)
        {
            if (targetSet)
            {
                _fileContents[vectorIndex].target = mat;
            }
            else
            {
                _fileContents[vectorIndex].input = mat;
            }
        }

        internal CsvFileDataSetReader Copy()
        {
            return new CsvFileDataSetReader(_setVariableIndexes, _dataSetInfo, _fileContents.Select(v => (v.input.Clone(), v.target.Clone())).ToArray(),_csvInternalReader);
        }

        public void Dispose()
        {
        }
    }
}