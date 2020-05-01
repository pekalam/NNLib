using System.Collections.Generic;

namespace NNLib
{
    public interface IReadOnlyNetwork
    {
        IReadOnlyList<Layer> ReadBaseLayers { get; }
    }

    public interface IReadOnlyNetwork<T> : IReadOnlyNetwork where T : Layer
    {
        IReadOnlyList<T> ReadLayers { get; }
    }
}