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
        private Matrix<double>[] _ignored;
        private readonly ICsvReader _csvReader;

        private CsvVectorReader(SupervisedSetVariableIndexes setVariableIndexes, DataSetInfo dataSetInfo, (Matrix<double> input, Matrix<double> target)[] fileContents, ICsvReader csvReader)
        {
            _dataSetInfo = dataSetInfo;
            CurrentIndexes = setVariableIndexes;
            _fileContents = fileContents;
            _csvReader = csvReader;
            _ignored = new Matrix<double>[_fileContents.Length];
        }

        public CsvVectorReader(ICsvReader csvReader,
            SupervisedSetVariableIndexes setVariableIndexes, DataSetInfo dataDataSetInfo)
        {
            CurrentIndexes = setVariableIndexes;
            _dataSetInfo = dataDataSetInfo;
            _csvReader = csvReader;
            _fileContents = csvReader.ReadVectorSets(dataDataSetInfo.FileParts, setVariableIndexes);
            _ignored = new Matrix<double>[_fileContents.Length];
        }

        public SupervisedSetVariableIndexes CurrentIndexes { get; private set; }

        internal void ChangeVariables(SupervisedSetVariableIndexes newVariableIndexes)
        {
            var newFileContents = new (Matrix<double> input, Matrix<double> target)[_fileContents.Length];
            var newIgnored = new Matrix<double>[_ignored.Length];

            //dst=0 -target dst=1 - input dst=2 - ignore
            var indexMap = new (int dst, int index)[newVariableIndexes.InputVarIndexes.Length +
                                   newVariableIndexes.TargetVarIndexes.Length + newVariableIndexes.Ignored.Length];

            for (int i = 0; i < newVariableIndexes.InputVarIndexes.Length; i++)
            {
                var ind = CurrentIndexes.InputVarIndexes.IndexOf(newVariableIndexes.InputVarIndexes[i]);
                if (ind != -1)
                {
                    indexMap[newVariableIndexes.InputVarIndexes[i]] = (1, ind);
                }

                ind = CurrentIndexes.TargetVarIndexes.IndexOf(newVariableIndexes.InputVarIndexes[i]);
                if (ind != -1)
                {
                    indexMap[newVariableIndexes.InputVarIndexes[i]] = (0, ind);
                }

                ind = CurrentIndexes.Ignored.IndexOf(newVariableIndexes.InputVarIndexes[i]);
                if (ind != -1)
                {
                    indexMap[newVariableIndexes.InputVarIndexes[i]] = (2, ind);
                }
            }

            for (int i = 0; i < newVariableIndexes.TargetVarIndexes.Length; i++)
            {
                var ind = CurrentIndexes.InputVarIndexes.IndexOf(newVariableIndexes.TargetVarIndexes[i]);
                if (ind != -1)
                {
                    indexMap[newVariableIndexes.TargetVarIndexes[i]] = (1, ind);
                }

                ind = CurrentIndexes.TargetVarIndexes.IndexOf(newVariableIndexes.TargetVarIndexes[i]);
                if (ind != -1)
                {
                    indexMap[newVariableIndexes.TargetVarIndexes[i]] = (0, ind);
                }

                ind = CurrentIndexes.Ignored.IndexOf(newVariableIndexes.TargetVarIndexes[i]);
                if (ind != -1)
                {
                    indexMap[newVariableIndexes.TargetVarIndexes[i]] = (2, ind);
                }
            }


            for (int i = 0; i < _fileContents.Length; i++)
            {
                var input = Matrix<double>.Build.Dense(newVariableIndexes.InputVarIndexes.Length, 1);
                var target = Matrix<double>.Build.Dense(newVariableIndexes.TargetVarIndexes.Length, 1);
                var ignored = Matrix<double>.Build.Dense(newVariableIndexes.Ignored.Length, 1);

                int r = 0;
                for (int j = 0; j < newVariableIndexes.InputVarIndexes.Length; j++)
                {
                    var map = indexMap[newVariableIndexes.InputVarIndexes[j]];

                    if (map.dst == 1)
                    {
                        input[r++, 0] = _fileContents[i].input[map.index, 0];
                    }
                    else if(map.dst == 0)
                    {
                        input[r++, 0] = _fileContents[i].target[map.index, 0];
                    }
                    else
                    {
                        input[r++, 0] = _ignored[i][map.index, 0];
                    }
                }

                r = 0;
                for (int j = 0; j < newVariableIndexes.TargetVarIndexes.Length; j++)
                {
                    var map = indexMap[newVariableIndexes.TargetVarIndexes[j]];

                    if (map.dst == 1)
                    {
                        target[r++, 0] = _fileContents[i].input[map.index, 0];
                    }
                    else if (map.dst == 0)
                    {
                        target[r++, 0] = _fileContents[i].target[map.index, 0];
                    }
                    else
                    {
                        target[r++, 0] = _ignored[i][map.index, 0];
                    }
                }


                r = 0;
                for (int j = 0; j < newVariableIndexes.Ignored.Length; j++)
                {
                    var map = indexMap[newVariableIndexes.Ignored[j]];

                    if (map.dst == 1)
                    {
                        ignored[r++, 0] = _fileContents[i].input[map.index, 0];
                    }
                    else if (map.dst == 0)
                    {
                        ignored[r++, 0] = _fileContents[i].target[map.index, 0];
                    }
                    else
                    {
                        ignored[r++, 0] = _ignored[i][map.index, 0];
                    }
                }

                newFileContents[i] = (input, target);
                newIgnored[i] = ignored;
            }

            _fileContents = newFileContents;
            _ignored = newIgnored;
            CurrentIndexes = newVariableIndexes;
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