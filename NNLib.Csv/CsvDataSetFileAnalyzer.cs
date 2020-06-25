using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Runtime.CompilerServices;
using CsvHelper;
using Serilog;

[assembly: InternalsVisibleTo("NNLib.Csv.Tests")]
namespace NNLib.Csv
{
    internal class CsvDataSetFileAnalyzer
    {
        //TODO filestream
        public int GetRowSizeApproximation(string fileName)
        {
            using var fs = File.OpenRead(fileName);
            using var rdr = new StreamReader(fs);
            using var csv = new CsvReader(rdr, CultureInfo.CurrentCulture);

            csv.Read();
            csv.ReadHeader();
            csv.Read();

            var colCount = csv.Context.HeaderRecord.Length;

            var sz = 0;
            for (var i = 0; i < colCount; i++)
            {
                if (csv.TryGetField<string>(i, out var str))
                {
                    try
                    {
                        var _ = Convert.ToDouble(str);
                    }
                    catch (Exception)
                    {
                        sz += 2 * str.Length;
                        continue;
                    }

                    sz += sizeof(double);
                }
            }

            Log.Logger.Debug("Approximate size of csv row: {sz} fileName: {file}", sz, fileName);
            return sz;
        }

        public string[] GetVariableNames(string fileName)
        {
            using var fs = File.OpenRead(fileName);
            using var rdr = new StreamReader(fs);
            using var csv = new CsvReader(rdr, CultureInfo.CurrentCulture);
            csv.Read();
            csv.ReadHeader();

            var headers = csv.Context.HeaderRecord;
            return headers;
        }

        public List<long> GetDataItemsNewLinePositions(string fileName)
        {
            (int linesCount, List<long> newLinePositions) = FileHelpers.CountLinesAndGetPositions(fileName);
            Log.Logger.Debug("Found {n} new lines in {fileName}", newLinePositions.Count, fileName);
            newLinePositions.RemoveAt(0);
            return newLinePositions;
        }
    }
}