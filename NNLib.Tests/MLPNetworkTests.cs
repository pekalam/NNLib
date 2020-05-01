using System;
using FluentAssertions;
using MathNet.Numerics.LinearAlgebra;
using NNLib;
using NNLib.ActivationFunction;
using Xunit;
using Xunit.Abstractions;

namespace UnitTests
{
    public class MLPNetworkTests
    {
        private readonly ITestOutputHelper output;

        public MLPNetworkTests(ITestOutputHelper output)
        {
            this.output = output;
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

            net.ReadOutput[0, 0].Should().Be(5);
            net.ReadOutput[1, 0].Should().Be(6);
            net.ReadOutput.ColumnCount.Should().Be(1);
            net.ReadOutput.RowCount.Should().Be(2);


            l.CalculateOutput(Matrix<double>.Build.DenseOfArray(new double[,] {{2}, {0}}));

            l.ReadOutput[0, 0].Should().Be(2);
            l.ReadOutput[1, 0].Should().Be(4);
            l.ReadOutput.ColumnCount.Should().Be(1);
            l.ReadOutput.RowCount.Should().Be(2);
        }


    }
}