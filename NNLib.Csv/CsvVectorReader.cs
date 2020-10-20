using System;
using System.Linq;
using System.Runtime.CompilerServices;
using MathNet.Numerics.LinearAlgebra;
using NNLib.Common;

[assembly: InternalsVisibleTo("Data.Tests")]
[assembly: InternalsVisibleTo("Infrastructure.Data.Tests")]

namespace NNLib.Csv
{
    internal class CsvVectorReader
    {
        private readonly DataSetInfo _dataSetInfo;
        private (Matrix<double> input, Matrix<double> target)[] _fileContents;
        private readonly ICsvReader _csvReader;

        private CsvVectorReader(SupervisedSetVariableIndexes setVariableIndexes, DataSetInfo dataSetInfo, (Matrix<double> input, Matrix<double> target)[] fileContents, ICsvReader csvReader)
        {
            _dataSetInfo = dataSetInfo;
            CurrentIndexes = setVariableIndexes;
            _fileContents = fileContents;
            _csvReader = csvReader;
        }

        public CsvVectorReader(ICsvReader csvReader,
            SupervisedSetVariableIndexes setVariableIndexes, DataSetInfo dataDataSetInfo)
        {
            CurrentIndexes = setVariableIndexes;
            _dataSetInfo = dataDataSetInfo;
            _csvReader = csvReader;
            _fileContents = csvReader.ReadVectorSets(dataDataSetInfo.FilePart, setVariableIndexes);
        }

        public SupervisedSetVariableIndexes CurrentIndexes { get; private set; }

        internal void ChangeVariables(SupervisedSetVariableIndexes newVariableIndexes)
        {
            CurrentIndexes = newVariableIndexes;
            _fileContents = _csvReader.ReadVectorSets(_dataSetInfo.FilePart, CurrentIndexes);
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

        internal CsvVectorReader Copy()
        {
            return new CsvVectorReader(CurrentIndexes, _dataSetInfo, _fileContents.Select(v => (v.input.Clone(), v.target.Clone())).ToArray(),_csvReader);
        }
    }
}