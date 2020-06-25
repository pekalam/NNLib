using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Globalization;
using System.IO;
using System.IO.MemoryMappedFiles;
using CsvHelper;
using MathNet.Numerics.LinearAlgebra;
using NNLib.Common;

namespace NNLib.Csv
{




    internal interface ICsvInternalReader : IDisposable
    {
        List<(Matrix<double> input, Matrix<double> target)> ReadVectorSets(in FilePart filePart, SupervisedSetVariableIndexes setVariableIndexes);
    }

    internal class CsvInternalReader : ICsvInternalReader
    {
        private readonly MemoryMappedFile _mmf;

        public CsvInternalReader(string fileName)
        {
            var fs = new FileStream(fileName, FileMode.Open, FileAccess.Read, FileShare.Read);
            _mmf = MemoryMappedFile.CreateFromFile(fs, null, 0, MemoryMappedFileAccess.Read, HandleInheritability.Inheritable, false);
        }

        private Matrix<double> ReadVector(CsvReader csv, in ImmutableArray<int> indices)
        {
            var vector = new double[indices.Length];
            int i = 0;
            foreach (var index in indices)
            {
                vector[i++] = csv.GetField<double>(index);
            }
            
            return Matrix<double>.Build.Dense(vector.Length, 1, vector);
        }

        public List<(Matrix<double> input, Matrix<double> target)> ReadVectorSets(in FilePart filePart, SupervisedSetVariableIndexes setVariableIndexes)
        {
            using var accessor = _mmf.CreateViewStream(filePart.Offset, filePart.End - filePart.Offset, MemoryMappedFileAccess.Read);
            using var rdr = new StreamReader(accessor);
            using var csv = new CsvReader(rdr, CultureInfo.CurrentCulture);

            if (filePart.Offset == 0)
            {
                csv.Read();
                csv.ReadHeader();
            }

            var i = 0;
            var vectorsSet = new List<(Matrix<double> input, Matrix<double> target)>();

            while (i < filePart.DataItems)
            {
                if (!csv.Read())
                {
                    throw new Exception("Csv reader error");
                }

                Matrix<double> input = ReadVector(csv, setVariableIndexes.InputVarIndexes);
                Matrix<double> target = ReadVector(csv, setVariableIndexes.TargetVarIndexes);
                vectorsSet.Add((input, target));
                i++;
            }

            return vectorsSet;
        }

        public void Dispose()
        {
            _mmf?.Dispose();
        }
    }
}