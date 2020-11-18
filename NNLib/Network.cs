using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using NNLib.Data;

[assembly: InternalsVisibleTo("DynamicProxyGenAssembly2")]
namespace NNLib
{
    public interface INetwork
    {
        event Action<INetwork> StructureChanged;
        IReadOnlyList<Layer> BaseLayers { get; }
        int TotalLayers { get; }
        int TotalNeurons { get; }
        int TotalSynapses { get; }
        int TotalBiases { get; }
    }

    public abstract class Network<T> : INetwork where T : Layer
    {
        protected readonly List<T> _layers;

        public event Action<INetwork> StructureChanged = null!;

        protected Network(params T[] layers)
        {
            if (layers.Length == 0)
            {
                throw new ArgumentException("Layers array must have length greater than 0");
            }
            ValidateLayersInputsAndOutputs(layers.ToList());

            _layers = new List<T>(layers);
            foreach (var layer in _layers)
            {
                layer.AssignNetwork(this);
                AssignEventHandlers(layer);
                if (!layer.IsInitialized)
                {
                    layer.Initialize();
                }
            }

            foreach (var layer in _layers)
            {
                layer.InitializeMemory();
            }
        }


        public Matrix<double>? Output;

        public IReadOnlyList<T> Layers => _layers;
        public IReadOnlyList<Layer> BaseLayers => _layers;
        public int TotalLayers => _layers.Count;
        public int TotalNeurons => _layers.Sum(layer => layer.NeuronsCount);
        public int TotalSynapses => _layers.Sum(l => l.InputsCount * l.NeuronsCount);
        public int TotalBiases => TotalNeurons;

        public virtual void InitializeMemoryForData(SupervisedTrainingSamples data)
        {
            foreach (var layer in _layers)
            {
                layer.InitializeMemoryForData(data);
            }
        }

        public void AddLayer() => AddLayer(CreateOutputLayer(_layers[^1].NeuronsCount, 1));

        public void AddLayer(T layer)
        {
            ValidateLayersInputsAndOutputs(_layers.Concat(new []{layer}).ToList());
            _layers.Add(layer);
            layer.AssignNetwork(this);
            AssignEventHandlers(layer);
            if (!layer.IsInitialized)
            {
                layer.Initialize();
            }
            layer.InitializeMemory();
            RaiseNetworkStructureChanged();
        }

        public void RemoveLayer(int ind)
        {
            Layer? next = ind == _layers.Count - 1 ? null : _layers[ind + 1];
            Layer? prev = ind == 0 ? null : _layers[ind - 1];

            next?.AdjustToMatchPrevious(prev);
            _layers.RemoveAt(ind);
            RaiseNetworkStructureChanged();
        }

        public void RemoveLayer(T layer)
        {
            var ind = _layers.IndexOf(layer);
            if (ind == -1)
            {
                throw new ArgumentException("Cannot find layer");
            }
            RemoveLayer(ind);
        }

        public T InsertAfter(int ind)
        {
            ind++;
            if (ind > TotalLayers || ind < 0) throw new ArgumentException("Cannot insert after " + ind + " - index out of bounds");

            Func<int, int, T> layerFunc;
            if (ind == TotalLayers)
            {
                layerFunc = CreateOutputLayer;
            }
            else
            {
                layerFunc = CreateHiddenLayer;
            }
            var layer = layerFunc(ind == 0 ? Layers[0].InputsCount : Layers[ind - 1].NeuronsCount,
                ind == TotalLayers ? Layers[^1].NeuronsCount : Layers[ind].InputsCount);

            _layers.Insert(ind, layer);
            layer.AssignNetwork(this);
            AssignEventHandlers(layer);
            if (!layer.IsInitialized)
            {
                layer.Initialize();
            }
            layer.InitializeMemory();
            RaiseNetworkStructureChanged();
            return layer;
        }

        public T InsertBefore(int ind) => InsertAfter(ind - 1);

        protected void AssignEventHandlers(Layer layer)
        {
            if (layer == _layers[0])
            {
                layer.InputsCountChanged += l =>
                {
                    RaiseNetworkStructureChanged();
                    l.InitializeMemory();
                };
            }
            layer.NeuronsCountChanged += LayerOnNeuronsCountChanged;
        }

        protected void RaiseNetworkStructureChanged() => StructureChanged?.Invoke(this);

        private void LayerOnNeuronsCountChanged(Layer layer)
        {
            for (int i = 0; i < _layers.Count; i++)
            {
                if (_layers[i] == layer)
                {
                    if (i + 1 < _layers.Count)
                    {
                        _layers[i + 1].InputsCount = layer.NeuronsCount;
                    }
                    break;
                }
            }
            RaiseNetworkStructureChanged();
            layer.InitializeMemory();
        }

        private void ValidateLayersInputsAndOutputs(List<T> layers)
        {
            if (layers[0].InputsCount == 0)
            {
                throw new ArgumentException("InputsCount cannot be equal to 0");
            }

            if (layers[^1].NeuronsCount == 0)
            {
                throw new ArgumentException("NeuronsCount cannot be equal to 0");
            }

            for (int i = 1; i < layers.Count; i++)
            {
                if (layers[i].InputsCount != layers[i - 1].NeuronsCount)
                {
                    throw new ArgumentException($"Inputs count of layer {i} does not match neuronsCount of layer {i-1}");
                }
            }
        }

        public void ResetParameters()
        {
            foreach (var layer in _layers)
            {
                layer.ResetParameters();
            }
        }

        internal abstract T CreateHiddenLayer(int inputsCount, int neuronsCount);
        internal abstract T CreateOutputLayer(int inputsCount, int neuronsCount);
        public abstract void CalculateOutput(Matrix<double> input);
    }
}