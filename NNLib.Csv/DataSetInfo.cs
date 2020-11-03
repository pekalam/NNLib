using System.Collections.Generic;
using System.Runtime.CompilerServices;
using NNLib.Common;
using NNLib.Data;

[assembly: InternalsVisibleTo("DynamicProxyGenAssembly2")]
namespace NNLib.Csv
{
    internal class DataSetInfo
    {
        public FilePart[] FileParts { get; }
        public int SetSize { get; }
        public int PageSize { get; }
        public DataSetType DataSetType { get; }
        public string[] VariableNames { get; }

        public DataSetInfo(FilePart[] fileParts, int size, int pageSize, DataSetType dataSetType, string[] variableNames)
        {
            FileParts = fileParts;
            SetSize = size;
            PageSize = pageSize;
            DataSetType = dataSetType;
            VariableNames = variableNames;
        }
    }
}