﻿using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using MathNet.Numerics.LinearAlgebra.Storage;

[assembly: InternalsVisibleTo("DynamicProxyGenAssembly2")]
namespace NNLib
{
    public interface INetwork
    {
        IReadOnlyList<Layer> BaseLayers { get; }
    }


    public abstract class Network<T> : INetwork where T : Layer
    {
        protected readonly List<T> _layers;

        public event Action<INetwork> StructureChanged;

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

        }


        public Matrix<double>? Output;

        public IReadOnlyList<T> Layers => _layers;
        public IReadOnlyList<Layer> BaseLayers => _layers;
        public int TotalLayers => _layers.Count;
        public int TotalNeurons => _layers.Sum(layer => layer.NeuronsCount);
        public int TotalSynapses => _layers.Sum(l => l.InputsCount * l.NeuronsCount);
        public int TotalBiases => TotalNeurons;

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
            RaiseNetworkStructureChanged();
        }

        public void RemoveLayer(T layer)
        {
            var ind = _layers.IndexOf(layer);
            if (ind == -1)
            {
                throw new ArgumentException("Cannot find layer");
            }

            Layer? next = ind == _layers.Count - 1 ? null : _layers[ind+1];
            Layer? prev = ind == 0 ? null : _layers[ind-1];

            next?.AdjustMatSize(prev);
            _layers.RemoveAt(ind);
            RaiseNetworkStructureChanged();
        }

        protected void AssignEventHandlers(Layer layer)
        {
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

        public abstract void CalculateOutput(Matrix<double> input);
    }
}