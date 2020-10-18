using System;
using System.Runtime.InteropServices.ComTypes;
using FluentAssertions;
using MathNet.Numerics.LinearAlgebra;
using NNLib.ActivationFunction;
using NNLib.MLP;
using Xunit;
using Xunit.Abstractions;

namespace NNLib.Tests
{
    public class MLPNetworkTests
    {
        private readonly ITestOutputHelper output;
        private const int DefaultNormal = 0;
        private const int Xavier = 1;
        private const int Normal = 2;

        public MLPNetworkTests(ITestOutputHelper output)
        {
            this.output = output;
        }

        private MatrixBuilder GetMatBuilder(int num)
        {
            return num switch
            {
                DefaultNormal => new DefaultNormDistMatrixBuilder(),
                Xavier => new XavierMatrixBuilder(),
                Normal => new NormDistMatrixBuilder(new NormDistMatrixBuilderOptions()
                {
                    WMean = 0, BMean = 0, WStdDev = 0.1, BStdDev = 0.01,
                }),
            _ => throw new Exception(),
            };
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

        [Theory]
        [InlineData(DefaultNormal)]
        [InlineData(Normal)]
        public void NeuronsCount_when_changed_neurons_count_in_layer_is_changed(int matbuilder)
        {
            var l = new PerceptronLayer(2, 2, new LinearActivationFunction(), GetMatBuilder(matbuilder));
            l.Initialize();
            l.NeuronsCount += 1;

            l.Weights.RowCount.Should().Be(3);
            l.Weights.ColumnCount.Should().Be(2);

            l.NeuronsCount -= 2;
            l.Weights.RowCount.Should().Be(1);
            l.Weights.ColumnCount.Should().Be(2);


            Assert.Throws<ArgumentException>(() => l.NeuronsCount -= 1);
        }


        [Theory]
        [InlineData(DefaultNormal)]
        [InlineData(Normal)]
        [InlineData(Xavier)]
        public void NeuronsCount_when_changed_neurons_count_in_layer_adjacent_layers_are_updated(int matbuilder)
        {
            var l = new PerceptronLayer(1, 2, new LinearActivationFunction(), GetMatBuilder(matbuilder));
            var l2 = new PerceptronLayer(2, 2, new LinearActivationFunction(), GetMatBuilder(matbuilder));
            var l3 = new PerceptronLayer(2, 2, new LinearActivationFunction(), GetMatBuilder(matbuilder));
            var net = new MLPNetwork(l, l2, l3);

            l2.NeuronsCount += 1;
            l3.InputsCount.Should().Be(3);
        }

        [Theory]
        [InlineData(DefaultNormal)]
        [InlineData(Normal)]
        public void NeuronsCount_when_changed_does_not_change_existing_weights_and_biases(int matbuilder)
        {
            var l = new PerceptronLayer(2, 2, new LinearActivationFunction(), GetMatBuilder(matbuilder));
            l.Initialize();
            var w1 = l.Weights.Clone();
            var b1 = l.Biases.Clone();
            l.NeuronsCount += 1;
            l.Weights.RowCount.Should().Be(3);

            if (matbuilder == 0)
            {
                for (int i = 0; i < w1.RowCount; i++)
                {
                    for (int j = 0; j < w1.ColumnCount; j++)
                    {
                        l.Weights[i, j].Should().Be(w1[i, j]);
                    }

                    l.Biases[i, 0].Should().Be(b1[i, 0]);
                }
            }


            l.NeuronsCount -= 2;
            l.Weights.RowCount.Should().Be(1);

            if (matbuilder == 0)
            {
                for (int i = 0; i < l.Weights.RowCount; i++)
                {
                    for (int j = 0; j < l.Weights.ColumnCount; j++)
                    {
                        l.Weights[i, j].Should().Be(w1[i, j]);
                    }
                    l.Biases[i, 0].Should().Be(b1[i, 0]);
                }
            }

        }

        [Theory]
        [InlineData(DefaultNormal)]
        [InlineData(Normal)]
        public void InputsCount_when_changed_does_not_change_existing_weights_and_biases(int matbuilder)
        {
            var l = new PerceptronLayer(2, 2, new LinearActivationFunction(), GetMatBuilder(matbuilder));
            l.Initialize();
            var w1 = l.Weights.Clone();
            var b1 = l.Biases.Clone();
            l.InputsCount += 1;
            l.Weights.ColumnCount.Should().Be(3);

            if (matbuilder == 0)
            {
                for (int i = 0; i < w1.RowCount; i++)
                {
                    for (int j = 0; j < w1.ColumnCount; j++)
                    {
                        l.Weights[i, j].Should().Be(w1[i, j]);
                    }

                    l.Biases[i, 0].Should().Be(b1[i, 0]);
                }
            }


            l.InputsCount -= 2;
            l.Weights.ColumnCount.Should().Be(1);

            if (matbuilder == 0)
            {
                for (int i = 0; i < l.Weights.RowCount; i++)
                {
                    for (int j = 0; j < l.Weights.ColumnCount; j++)
                    {
                        l.Weights[i, j].Should().Be(w1[i, j]);
                    }

                    l.Biases[i, 0].Should().Be(b1[i, 0]);
                }
            }
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

        [Theory]
        [InlineData(DefaultNormal)]
        [InlineData(Xavier)]
        public void Clone_creates_deep_copy(int matbuilder)
        {
            var l = new PerceptronLayer(1, 2, new LinearActivationFunction(), GetMatBuilder(matbuilder));
            var l2 = new PerceptronLayer(2, 2, new LinearActivationFunction(), GetMatBuilder(matbuilder));
            var l3 = new PerceptronLayer(2, 1, new LinearActivationFunction(), GetMatBuilder(matbuilder));
            var net = new MLPNetwork(l, l2, l3);

            var input = Matrix<double>.Build.Dense(1, 1);
            input[0, 0] = 1;

            net.CalculateOutput(input);

            var net2 = net.Clone();
            net2.TotalLayers.Should().Be(net.TotalLayers);

            for (int i = 0; i < net.TotalLayers; i++)
            {
                net2.Layers[i].Should().BeEquivalentTo(net.Layers[i], opt =>
                {
                   return opt.Excluding(p => p.Network);
                });
                net2.Layers[i].Network.Should().NotBe(net.Layers[i].Network);
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
                net.Layers[i].Biases.CompareTo(net2.Layers[i].Biases).Should().BeTrue();
                net.Layers[i].Output.CompareTo(net2.Layers[i].Output).Should().BeFalse();
            }
        }


        [Theory]
        [InlineData(DefaultNormal)]
        [InlineData(Xavier)]
        public void RemoveLayer_removes_hidden_layer(int matbuilder)
        {
            var l = new PerceptronLayer(1, 8, new LinearActivationFunction(), GetMatBuilder(matbuilder));
            var l2 = new PerceptronLayer(8, 2, new LinearActivationFunction(), GetMatBuilder(matbuilder));
            var l3 = new PerceptronLayer(2, 1, new LinearActivationFunction(), GetMatBuilder(matbuilder));
            var net = new MLPNetwork(l, l2, l3);

            net.RemoveLayer(l2);

            l.NeuronsCount.Should().Be(8);
            l.InputsCount.Should().Be(1);
            l3.InputsCount.Should().Be(8);
            l3.NeuronsCount.Should().Be(1);
            net.TotalLayers.Should().Be(2);
        }

        [Theory]
        [InlineData(DefaultNormal)]
        [InlineData(Xavier)]
        public void RemoveLayer_removes_output_layer(int matbuilder)
        {
            var l = new PerceptronLayer(1, 8, new LinearActivationFunction(), GetMatBuilder(matbuilder));
            var l2 = new PerceptronLayer(8, 2, new LinearActivationFunction(), GetMatBuilder(matbuilder));
            var l3 = new PerceptronLayer(2, 1, new LinearActivationFunction(), GetMatBuilder(matbuilder));
            var net = new MLPNetwork(l, l2, l3);

            net.RemoveLayer(l3);

            l.NeuronsCount.Should().Be(8);
            l.InputsCount.Should().Be(1);
            l2.InputsCount.Should().Be(8);
            l2.NeuronsCount.Should().Be(2);
            net.TotalLayers.Should().Be(2);
        }

        [Theory]
        [InlineData(DefaultNormal)]
        [InlineData(Xavier)]
        public void RemoveLayer_removes_input_layer(int matbuilder)
        {
            var l = new PerceptronLayer(1, 8, new LinearActivationFunction(), GetMatBuilder(matbuilder));
            var l2 = new PerceptronLayer(8, 2, new LinearActivationFunction(), GetMatBuilder(matbuilder));
            var l3 = new PerceptronLayer(2, 1, new LinearActivationFunction(), GetMatBuilder(matbuilder));
            var net = new MLPNetwork(l, l2, l3);

            net.RemoveLayer(l);

            l2.NeuronsCount.Should().Be(2);
            l2.InputsCount.Should().Be(8);
            l3.InputsCount.Should().Be(2);
            l3.NeuronsCount.Should().Be(1);
            net.TotalLayers.Should().Be(2);
        }

        [Theory]
        [InlineData(DefaultNormal)]
        [InlineData(Normal)]
        public void RebuildMatrices_creates_new_matrices(int matbuilder)
        {
            var l = new PerceptronLayer(1, 8, new LinearActivationFunction(), GetMatBuilder(matbuilder));
            l.Initialize();

            var w1 = l.Weights.Clone();
            var b1 = l.Biases.Clone();
            
            l.ResetParameters();

            for (int i = 0; i < w1.RowCount; i++)
            {
                for (int j = 0; j < w1.ColumnCount; j++)
                {
                    l.Weights[i, j].Should().NotBe(w1[i, j]);
                }

            }
        }

        [Theory]
        [InlineData(DefaultNormal)]
        [InlineData(Normal)]
        [InlineData(Xavier)]
        public void InsertAfter_inserts_new_layer_after_given_index(int matbuilder)
        {
            var l = new PerceptronLayer(1, 8, new LinearActivationFunction(), GetMatBuilder(matbuilder));
            var l2 = new PerceptronLayer(8, 2, new LinearActivationFunction(), GetMatBuilder(matbuilder));
            var l3 = new PerceptronLayer(2, 1, new LinearActivationFunction(), GetMatBuilder(matbuilder));
            var net = new MLPNetwork(l, l2, l3);

            var toInsert = net.InsertAfter(2);

            toInsert.IsInitialized.Should().BeTrue();
            toInsert.InputsCount.Should().Be(1);
            toInsert.NeuronsCount.Should().Be(1);
            net.Layers[^1].Should().BeSameAs(toInsert);

            toInsert=net.InsertAfter(0);
            toInsert.IsInitialized.Should().BeTrue();
            toInsert.InputsCount.Should().Be(8);
            toInsert.NeuronsCount.Should().Be(8);
            net.Layers[1].Should().BeSameAs(toInsert);

        }


        [Theory]
        [InlineData(DefaultNormal)]
        [InlineData(Xavier)]
        public void InsertBefore_inserts_new_layer_before_given_index(int matbuilder)
        {
            var l = new PerceptronLayer(1, 8, new LinearActivationFunction(), GetMatBuilder(matbuilder));
            var l2 = new PerceptronLayer(8, 2, new LinearActivationFunction(), GetMatBuilder(matbuilder));
            var l3 = new PerceptronLayer(2, 1, new LinearActivationFunction(), GetMatBuilder(matbuilder));
            var net = new MLPNetwork(l, l2, l3);

            var toInsert = net.InsertBefore(2);

            toInsert.IsInitialized.Should().BeTrue();
            toInsert.NeuronsCount.Should().Be(2);
            toInsert.InputsCount.Should().Be(2);
            net.Layers[^2].Should().BeSameAs(toInsert);
            
            toInsert=net.InsertBefore(0);
            toInsert.IsInitialized.Should().BeTrue();
            toInsert.NeuronsCount.Should().Be(1);
            toInsert.InputsCount.Should().Be(1);
            net.Layers[0].Should().BeSameAs(toInsert);
        }
    }
}
