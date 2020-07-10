using System;
using System.Runtime.CompilerServices;
using MathNet.Numerics.LinearAlgebra;
using NNLib.Common;

[assembly: InternalsVisibleTo("Infrastructure.Data.Tests")]
namespace NNLib.Csv
{
    internal class CsvFileVectorSet : IVectorSet
    {
        internal readonly CsvFileDataSetReader FileReader;
        private readonly bool _targetSet;

        //Todo count
        public CsvFileVectorSet(CsvFileDataSetReader fileReader, int vectorsCount, bool targetSet)
        {
            if (vectorsCount <= 0)
            {
                throw new ArgumentException("Count cannot be le 0");
            }
            FileReader = fileReader;
            _targetSet = targetSet;
            Count = vectorsCount;
        }

        public Matrix<double> this[int index]
        {
            get => FileReader.ReadAt(_targetSet, index);
            set => FileReader.WriteAt(_targetSet, index, value);
        }

        public CsvFileVectorSet Copy(CsvFileDataSetReader fileReader)
        {
            return new CsvFileVectorSet(fileReader, Count, _targetSet);
        }

        public int Count { get; }

        public void Dispose()
        {
            FileReader?.Dispose();
        }
    }
}