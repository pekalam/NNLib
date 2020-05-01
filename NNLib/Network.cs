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

    public abstract class Network<T> : IReadOnlyNetwork<T> where T : Layer
    {
        internal List<T> Layers;
        internal Matrix<double> Output;

        protected Network(params T[] layers)
        {
            if (layers.Length == 0)
            {
                throw new ArgumentException();
            }
            ValidateLayersInputsAndOutputs(layers);

            Layers = new List<T>(layers);
            foreach (var layer in Layers)
            {
                layer.AssignNetwork(this);
                AssignEventHandlers(layer);
            }
        }

        public IReadOnlyList<T> ReadLayers => Layers.AsReadOnly();
        public IReadOnlyList<Layer> ReadBaseLayers => Layers.AsReadOnly();
        public ReadMatrixWrapper ReadOutput => Output;
        public int TotalLayers => Layers.Count;
        public int TotalNeurons => Layers.Sum(layer => layer.NeuronsCount);
        public bool Locked { get; private set; }


        private void CheckIsLocked(){if(Locked) throw new ObjectLockedException("Network locked");}

        internal virtual void Lock([CallerMemberName] string caller = "")
        {
            CheckIsLocked();
            foreach (var layer in Layers)
            {
                layer.Lock();
            }
            Trace.WriteLine("Network obj LOCKED by " + caller);
            Locked = true;
        }

        internal virtual void Unlock([CallerMemberName] string caller = "")
        {
            foreach (var layer in Layers)
            {
                layer.Unlock();
            }
            Trace.WriteLine("Network obj UNLOCKED by " + caller);
            Locked = false;
        }

        public void AddLayer(T layer)
        {
            CheckIsLocked();

            ValidateLayersInputsAndOutputs(Layers.Concat(new []{layer}).ToArray());
            Layers.Add(layer);
            layer.AssignNetwork(this);
            AssignEventHandlers(layer);
        }

        private void AssignEventHandlers(Layer layer)
        {
            layer.NeuronsCountChanged += LayerOnNeuronsCountChanged;
        }

        private void LayerOnNeuronsCountChanged(Layer layer)
        {
            for (int i = 0; i < Layers.Count; i++)
            {
                if (Layers[i] == layer)
                {
                    if (i + 1 < Layers.Count)
                    {
                        Layers[i + 1].InputsCount = layer.NeuronsCount;
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