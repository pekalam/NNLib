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
            List<long> newLinePositions = FileHelpers.CountLinesAndGetPositions(fileName);
            Log.Logger.Debug("Found {n} new lines in {fileName}", newLinePositions.Count, fileName);
            newLinePositions.RemoveAt(0);
            return newLinePositions;
        }
    }
}