using System.Collections.Generic;
using System.Runtime.CompilerServices;
using NNLib.Common;

[assembly: InternalsVisibleTo("DynamicProxyGenAssembly2")]
namespace NNLib.Csv
{
    internal class DataSetInfo
    {
        private List<FilePart> _fileParts;

        public IReadOnlyList<FilePart> FileParts => _fileParts;
        public int SetSize { get; }
        public int PageSize { get; }
        public DataSetType DataSetType { get; }
        public string[] VariableNames { get; }

        public DataSetInfo(List<FilePart> fileParts, int size, int pageSize, DataSetType dataSetType, string[] variableNames)
        {
            _fileParts = fileParts;
            SetSize = size;
            PageSize = pageSize;
            DataSetType = dataSetType;
            VariableNames = variableNames;
        }
    }
}