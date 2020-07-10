using System.Collections.Generic;
using System.Runtime.CompilerServices;
using NNLib.Common;

[assembly: InternalsVisibleTo("DynamicProxyGenAssembly2")]
namespace NNLib.Csv
{
    internal class DataSetInfo
    {
        public FilePart FilePart { get; }
        public int SetSize { get; }
        public int PageSize { get; }
        public DataSetType DataSetType { get; }
        public string[] VariableNames { get; }

        public DataSetInfo(FilePart filePart, int size, int pageSize, DataSetType dataSetType, string[] variableNames)
        {
            FilePart = filePart;
            SetSize = size;
            PageSize = pageSize;
            DataSetType = dataSetType;
            VariableNames = variableNames;
        }
    }
}