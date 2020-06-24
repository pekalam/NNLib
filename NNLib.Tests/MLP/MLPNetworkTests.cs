using System;
using FluentAssertions;
using MathNet.Numerics.LinearAlgebra;
using NNLib.ActivationFunction;
using Xunit;
using Xunit.Abstractions;

namespace NNLib.Tests
{
    public class MLPNetworkTests
    {
        private readonly ITestOutputHelper output;

        public MLPNetworkTests(ITestOutputHelper output)
        {
            this.output = output;
        }

        [Fact]
        public void MLPNetwork_when_constructed_has_valid_props()
        {
            var l = new PerceptronLayer(1, 2, new LinearActivationFunction());
            var l2 = new PerceptronLayer(2, 2, new LinearActivationFunction());
            var l3 = new PerceptronLayer(2, 3, new LinearActivationFunction());
            var net = new MLPNetwork(l, l2, l3);

            net.TotalLayers.Should().Be(3);
            net.TotalBiases.Should().Be(7);
            net.TotalNeurons.Should().Be(7);
            net.TotalSynapses.Should().Be(2 + 4 + 6);
        }

        [Fact]
        public void NeuronsCount_when_changed_neurons_count_in_layer_is_changed()
        {
            var l = new PerceptronLayer(2, 2, new LinearActivationFunction());
            l.NeuronsCount += 1;

            l.Weights.RowCount.Should().Be(3);
            l.Weights.ColumnCount.Should().Be(2);

            l.NeuronsCount -= 2;
            l.Weights.RowCount.Should().Be(1);
            l.Weights.ColumnCount.Should().Be(2);


            Assert.Throws<ArgumentException>(() => l.NeuronsCount -= 1);
        }


        [Fact]
        public void NeuronsCount_when_changed_neurons_count_in_layer_adjacent_layers_are_updated()
        {
            var l = new PerceptronLayer(1, 2, new LinearActivationFunction());
            var l2 = new PerceptronLayer(2, 2, new LinearActivationFunction());
            var l3 = new PerceptronLayer(2, 2, new LinearActivationFunction());
            var net = new MLPNetwork(l, l2, l3);

            l2.NeuronsCount += 1;
            l3.InputsCount.Should().Be(3);
        }


        [Fact]
        public void CalculateOutput_calculates_valid_output()
        {
            var l = new PerceptronLayer(2, 2, new LinearActivationFunction());
            var net = new MLPNetwork(l);

            l.Weights[0, 0] = 1;
            l.Weights[0, 1] = 2;

            l.Weights[1, 0] = 2;
            l.Weights[1, 1] = 2;

            l.Biases[0, 0] = 0;
            l.Biases[1, 0] = 0;


            net.CalculateOutput(Matrix<double>.Build.DenseOfArray(new double[,] {{1}, {2}}));

            net.Output[0, 0].Should().Be(5);
            net.Output[1, 0].Should().Be(6);
            net.Output.ColumnCount.Should().Be(1);
            net.Output.RowCount.Should().Be(2);


            l.CalculateOutput(Matrix<double>.Build.DenseOfArray(new double[,] {{2}, {0}}));

            l.Output[0, 0].Should().Be(2);
            l.Output[1, 0].Should().Be(4);
            l.Output.ColumnCount.Should().Be(1);
            l.Output.RowCount.Should().Be(2);
        }

        [Fact]
        public void Clone_creates_deep_copy()
        {
            var l = new PerceptronLayer(1, 2, new LinearActivationFunction());
            var l2 = new PerceptronLayer(2, 2, new LinearActivationFunction());
            var l3 = new PerceptronLayer(2, 1, new LinearActivationFunction());
            var net = new MLPNetwork(l, l2, l3);

            var input = Matrix<double>.Build.Dense(1, 1);
            input[0, 0] = 1;

            net.CalculateOutput(input);

            var net2 = net.Clone();
            net2.TotalLayers.Should().Be(net.TotalLayers);

            for (int i = 0; i < net.TotalLayers; i++)
            {
                net.Layers[i].Weights.CompareTo(net2.Layers[i].Weights).Should().BeTrue();
                net.Layers[i].Biases.CompareTo(net2.Layers[i].Biases).Should().BeTrue();
                net.Layers[i].Output.CompareTo(net2.Layers[i].Output).Should().BeTrue();
            }

            input[0, 0] = 10;
            net2.Layers[0].Weights.Multiply(4, net2.Layers[0].Weights); 
            net2.Layers[1].Weights.Multiply(4, net2.Layers[1].Weights);
            net2.Layers[2].Weights.Multiply(4, net2.Layers[2].Weights);
            net2.Layers[0].Biases.Multiply(4, net2.Layers[0].Biases);
            net2.Layers[1].Biases.Multiply(4, net2.Layers[1].Biases);
            net2.Layers[2].Biases.Multiply(4, net2.Layers[2].Biases);
            net2.CalculateOutput(input);

            net2.TotalLayers.Should().Be(net.TotalLayers);

            for (int i = 0; i < net.TotalLayers; i++)
            {
                net.Layers[i].Weights.CompareTo(net2.Layers[i].Weights).Should().BeFalse();
                net.Layers[i].Biases.CompareTo(net2.Layers[i].Biases).Should().BeFalse();
                net.Layers[i].Output.CompareTo(net2.Layers[i].Output).Should().BeFalse();
            }
        }


        [Fact]
        public void RemoveLayer_removes_hidden_layer()
        {
            var l = new PerceptronLayer(1, 8, new LinearActivationFunction());
            var l2 = new PerceptronLayer(8, 2, new LinearActivationFunction());
            var l3 = new PerceptronLayer(2, 1, new LinearActivationFunction());
            var net = new MLPNetwork(l, l2, l3);

            net.RemoveLayer(l2);

            l.NeuronsCount.Should().Be(8);
            l.InputsCount.Should().Be(1);
            l3.InputsCount.Should().Be(8);
            l3.NeuronsCount.Should().Be(1);
            net.TotalLayers.Should().Be(2);
        }

        [Fact]
        public void RemoveLayer_removes_output_layer()
        {
            var l = new PerceptronLayer(1, 8, new LinearActivationFunction());
            var l2 = new PerceptronLayer(8, 2, new LinearActivationFunction());
            var l3 = new PerceptronLayer(2, 1, new LinearActivationFunction());
            var net = new MLPNetwork(l, l2, l3);

            net.RemoveLayer(l3);

            l.NeuronsCount.Should().Be(8);
            l.InputsCount.Should().Be(1);
            l2.InputsCount.Should().Be(8);
            l2.NeuronsCount.Should().Be(2);
            net.TotalLayers.Should().Be(2);
        }

        [Fact]
        public void RemoveLayer_removes_input_layer()
        {
            var l = new PerceptronLayer(1, 8, new LinearActivationFunction());
            var l2 = new PerceptronLayer(8, 2, new LinearActivationFunction());
            var l3 = new PerceptronLayer(2, 1, new LinearActivationFunction());
            var net = new MLPNetwork(l, l2, l3);

            net.RemoveLayer(l);

            l2.NeuronsCount.Should().Be(2);
            l2.InputsCount.Should().Be(8);
            l3.InputsCount.Should().Be(2);
            l3.NeuronsCount.Should().Be(1);
            net.TotalLayers.Should().Be(2);
        }
    }
}