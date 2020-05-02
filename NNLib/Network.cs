using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using MathNet.Numerics.LinearAlgebra;

[assembly: InternalsVisibleTo("DynamicProxyGenAssembly2")]
namespace NNLib
{
    public class ObjectLockedException : Exception
    {
        public ObjectLockedException()
        {
        }

        public ObjectLockedException(string message) : base(message)
        {
        }
    }

    public abstract class Network<T> : Lockable<T>, INetwork<T> where T : Layer
    {
        private readonly List<T> _layers;

        protected Network(params T[] layers)
        {
            if (layers.Length == 0)
            {
                throw new ArgumentException();
            }
            ValidateLayersInputsAndOutputs(layers);

            _layers = new List<T>(layers);
            foreach (var layer in _layers)
            {
                layer.AssignNetwork(this);
                AssignEventHandlers(layer);
            }

            SetLockableChildren(_layers);
        }


        public Matrix<double>? Output;

        public IReadOnlyList<T> Layers => _layers;
        public IReadOnlyList<Layer> BaseLayers => _layers;
        public int TotalLayers => _layers.Count;
        public int TotalNeurons => _layers.Sum(layer => layer.NeuronsCount);

        public void AddLayer(T layer)
        {
            CheckIsLocked();

            ValidateLayersInputsAndOutputs(_layers.Concat(new []{layer}).ToArray());
            _layers.Add(layer);
            layer.AssignNetwork(this);
            AssignEventHandlers(layer);
        }

        private void AssignEventHandlers(Layer layer)
        {
            layer.NeuronsCountChanged += LayerOnNeuronsCountChanged;
        }

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
        }

        private void ValidateLayersInputsAndOutputs(T[] layers)
        {
            if (layers[0].InputsCount == 0)
            {
                throw new ArgumentException("InputsCount cannot be equal to 0");
            }

            if (layers[^1].NeuronsCount == 0)
            {
                throw new ArgumentException("NeuronsCount cannot be equal to 0");
            }

            for (int i = 1; i < layers.Length; i++)
            {
                if (layers[i].InputsCount != layers[i - 1].NeuronsCount)
                {
                    throw new ArgumentException($"Inputs count of layer {i} does not match neuronsCount of layer {i-1}");
                }
            }
        }

        public abstract void CalculateOutput(Matrix<double> input);
    }
}