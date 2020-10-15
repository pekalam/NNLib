using System;
using System.Collections.Immutable;
using System.Globalization;
using System.IO;
using CsvHelper;
using MathNet.Numerics.LinearAlgebra;
using NNLib.Common;

namespace NNLib.Csv
{
    internal interface ICsvReader
    {
        (Matrix<double> input, Matrix<double> target)[] ReadVectorSets(in FilePart filePart,
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

        public (Matrix<double> input, Matrix<double> target)[] ReadVectorSets(in FilePart filePart,
            SupervisedSetVariableIndexes setVariableIndexes)
        {
            using var fs = new FileStream(_fileName, FileMode.Open, FileAccess.Read, FileShare.Read);
            fs.Seek(filePart.Offset, SeekOrigin.Begin);
            using var rdr = new StreamReader(fs);
            using var csv = new CsvHelper.CsvReader(rdr, CultureInfo.CurrentCulture);

            if (filePart.Offset == 0)
            {
                csv.Read();
                csv.ReadHeader();
            }

            var i = 0;
            var vectorsSet = new (Matrix<double> input, Matrix<double> target)[filePart.DataItems];

            while (i < filePart.DataItems)
            {
                if (!csv.Read())
                {
                    throw new Exception("Csv reader error");
                }

                Matrix<double> input = ReadVector(csv, setVariableIndexes.InputVarIndexes);
                Matrix<double> target = ReadVector(csv, setVariableIndexes.TargetVarIndexes);
                vectorsSet[i] = (input, target);
                i++;
            }

            return vectorsSet;
        }
    }
}