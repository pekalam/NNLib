using System.Collections.Generic;

namespace NNLib
{
    public interface INetwork
    {
        IReadOnlyList<Layer> BaseLayers { get; }
    }

    public interface INetwork<T> : INetwork where T : Layer
    {
        IReadOnlyList<T> Layers { get; }
        int TotalLayers { get; }
        int TotalNeurons { get; }
    }
}