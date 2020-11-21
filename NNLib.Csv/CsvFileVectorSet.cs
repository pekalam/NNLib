using MathNet.Numerics.LinearAlgebra;
using NNLib.Data;
using System;
using System.Runtime.CompilerServices;

[assembly: InternalsVisibleTo("Infrastructure.Data.Tests")]
namespace NNLib.Csv
{
    internal class CsvFileVectorSet : IVectorSet
    {
        internal readonly CsvVectorReader FileReader;
        private readonly bool _targetSet;

        public CsvFileVectorSet(CsvVectorReader fileReader, int vectorsCount, bool targetSet)
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

        public CsvFileVectorSet Copy(CsvVectorReader fileReader)
        {
            return new CsvFileVectorSet(fileReader, Count, _targetSet);
        }

        public int Count { get; }
        public event Action Modified = null!;


        internal void RaiseModified()
        {
            Modified?.Invoke();
        }

        public void Dispose()
        {
        }
    }
}