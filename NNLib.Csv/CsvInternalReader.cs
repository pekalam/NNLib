using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Immutable;
using System.Globalization;
using System.IO;
using System.Linq;

namespace NNLib.Csv
{
    internal class CsvReaderResult
    {
        public (Matrix<double> input, Matrix<double> target)[] VectorSets { get; set; }
        public Matrix<double>?[] Ignored { get; set; }
    }

    internal interface ICsvReader
    {
        CsvReaderResult ReadFile(FilePart[] fileParts,
            SupervisedSetVariableIndexes setVariableIndexes);
    }

    internal class CsvReader : ICsvReader
    {
        private readonly string _fileName;

        public CsvReader(string fileName)
        {
            _fileName = fileName;
        }

        private Matrix<double> ReadVector(CsvHelper.CsvReader csv, in ImmutableArray<int> indices)
        {
            var vector = new double[indices.Length];
            int i = 0;
            foreach (var index in indices)
            {
                vector[i++] = csv.GetField<double>(index);
            }

            return Matrix<double>.Build.Dense(vector.Length, 1, vector);
        }

        public CsvReaderResult ReadFile(FilePart[] fileParts,
            SupervisedSetVariableIndexes setVariableIndexes)
        {
            var dataItemsCount = fileParts.Sum(v => v.DataItems);
            var vectorsSet = new (Matrix<double> input, Matrix<double> target)[dataItemsCount];
            var ignored = new Matrix<double>?[dataItemsCount];
            var s = 0;


            foreach (var filePart in fileParts)
            {
                var i = 0;

                using var fs = new FileStream(_fileName, FileMode.Open, FileAccess.Read, FileShare.Read);
                fs.Seek(filePart.Offset, SeekOrigin.Begin);
                using var rdr = new StreamReader(fs);
                using var csv = new CsvHelper.CsvReader(rdr, CultureInfo.InvariantCulture);

                if (filePart.Offset == 0)
                {
                    csv.Read();
                    csv.ReadHeader();
                }


                while (i < filePart.DataItems)
                {
                    if (!csv.Read())
                    {
                        throw new Exception("Csv reader error");
                    }

                    Matrix<double> input = ReadVector(csv, setVariableIndexes.InputVarIndexes);
                    Matrix<double> target = ReadVector(csv, setVariableIndexes.TargetVarIndexes);
                    if (setVariableIndexes.Ignored.Length > 0)
                    {
                        var ignoredVec = ReadVector(csv, setVariableIndexes.Ignored);
                        ignored[s] = ignoredVec;
                    }
                    vectorsSet[s++] = (input, target);

                    i++;
                }
            }


            return new CsvReaderResult()
            {
                VectorSets = vectorsSet, Ignored = ignored,
            };
        }
    }
}