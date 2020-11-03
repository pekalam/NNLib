using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using FluentAssertions;
using NNLib.ActivationFunction;
using NNLib.MLP;
using NNLib.Training.LevenbergMarquardt;
using Xunit;

namespace NNLib.Tests.Training
{
    public class JacobianTests
    {
        [Fact]
        public void When_network_structure_is_changed_jacobian_can_be_calculated_for_new_network_structure()
        {
            var sets = TrainingTestUtils.AndGateSet();
            var net = TrainingTestUtils.CreateNetwork(2, (2, new SigmoidActivationFunction()), (1, new TanHActivationFunction()));
            var jacobian = new Jacobian(net, sets.Input);

            jacobian.CalcJacobian().J.ColumnCount.Should().Be(9);

            net.Layers[0].NeuronsCount++;

            jacobian.CalcJacobian().J.ColumnCount.Should().Be(13);

            net.AddLayer(new PerceptronLayer(1, 1, new LinearActivationFunction()));

            jacobian.CalcJacobian().J.ColumnCount.Should().Be(15);

            net.RemoveLayer(net.Layers[^1]);

            jacobian.CalcJacobian().J.ColumnCount.Should().Be(13);

            net.InsertAfter(net.Layers.Count - 1);

            jacobian.CalcJacobian().J.ColumnCount.Should().Be(15);
        }
    }
}
