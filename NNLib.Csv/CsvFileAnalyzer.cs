using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Runtime.CompilerServices;

[assembly: InternalsVisibleTo("NNLib.Csv.Tests")]
namespace NNLib.Csv
{
    internal class CsvFileAnalyzer
    {
        public string[] GetVariableNames(string fileName)
        {
            using var fs = File.OpenRead(fileName);
            using var rdr = new StreamReader(fs);
            using var csv = new CsvHelper.CsvReader(rdr, CultureInfo.InvariantCulture);
            csv.Read();
            csv.ReadHeader();

            var headers = csv.Context.HeaderRecord;
            return headers;
        }

        public List<long> GetDataItemsNewLinePositions(string fileName)
        {
            List<long> newLinePositions = FileHelpers.CountLinesAndGetPositions(fileName);
            Debug.WriteLine("Found {0} new lines in {1}", newLinePositions.Count, fileName);
            newLinePositions.RemoveAt(0);
            return newLinePositions;
        }
    }
}