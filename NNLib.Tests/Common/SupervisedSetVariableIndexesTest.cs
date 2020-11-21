using System;
using FluentAssertions;
using NNLib.Common;
using Xunit;

namespace NNLib.Tests.Common
{
    public class SupervisedSetVariableIndexesTest
    {
        [Fact]
        public void Ctor_when_unsorted_params_sorts_indexes()
        {
            var variables = new SupervisedSetVariableIndexes(new[] { 4, 3, 2, 1 }, new[] { 9, 10 });

            variables.InputVarIndexes.Should().BeInAscendingOrder();
            variables.TargetVarIndexes.Should().BeInAscendingOrder();
        }

        [Fact]
        public void Ctor_when_empty_input_throws()
        {
            new SupervisedSetVariableIndexes(new int[0], new[] {9, 10}).Error.Should().NotBeNull();
        }

        [Fact]
        public void Ctor_when_empty_target_throws()
        {
            new SupervisedSetVariableIndexes(new[] { 0 }, new int[0]).Error.Should().NotBeNull();
        }

        [Fact]
        public void Ctor_when_empty_target_and_input_throws()
        {
            new SupervisedSetVariableIndexes(new int[0], new int[0]).Error.Should().NotBeNull();
        }

        [Fact]
        public void Ctor_when_expect_and_input_contains_the_same_vars_throws()
        {
            new SupervisedSetVariableIndexes(new[] { 0, 1, 9 }, new[] { 0, 1, 2 }).Should().NotBeNull();
        }

        [Fact]
        public void IgnoreVariable_removes_indexes_from_target_and_input()
        {
            var orgvars = new SupervisedSetVariableIndexes(new[] { 1, 2, 3 }, new[] { 0, 4 });

            var vars = orgvars.ChangeVariableUse(0, VariableUses.Ignore);
            vars.InputVarIndexes.Should().BeEquivalentTo(1, 2, 3);
            vars.TargetVarIndexes.Should().BeEquivalentTo(4);
            vars.Ignored.Should().BeEquivalentTo(0);

            vars = vars.ChangeVariableUse(1, VariableUses.Ignore);
            vars.InputVarIndexes.Should().BeEquivalentTo(2, 3);
            vars.TargetVarIndexes.Should().BeEquivalentTo(4);
            vars.Ignored.Should().BeEquivalentTo(0, 1);

            orgvars.Ignored.Length.Should().Be(0);
            orgvars.InputVarIndexes.Should().BeEquivalentTo(1, 2, 3);
            orgvars.TargetVarIndexes.Should().BeEquivalentTo(0, 4);
        }

        [Fact]
        public void IgnoreVariable_when_already_ignored_returns_vars()
        {
            var orgvars = new SupervisedSetVariableIndexes(new[] { 1, 2, 3 }, new[] { 0, 4 });
            var vars = orgvars.ChangeVariableUse(0, VariableUses.Ignore);
            var vars2 = vars.ChangeVariableUse(0, VariableUses.Ignore);

            vars.Ignored.Should().BeEquivalentTo(0);
            vars2.Ignored.Should().BeEquivalentTo(0);
        }

        [Fact]
        public void IgnoreVariable_when_contains_only_one_var_throws()
        {
            var vars = new SupervisedSetVariableIndexes(new[] { 1 }, new[] { 0 });

            vars.ChangeVariableUse(0, VariableUses.Ignore).Error.Should().NotBeNull();
            vars.ChangeVariableUse(1, VariableUses.Ignore).Error.Should().NotBeNull();
        }

        [Fact]
        public void SwapUse_when_valid_params_swaps_input_and_target_elements()
        {
            var vars = new SupervisedSetVariableIndexes(new[] { 1, 2, 3 }, new[] { 0 });

            var newVars = vars.ChangeVariableUse(2, VariableUses.Target);

            newVars.InputVarIndexes.Should().BeEquivalentTo(1, 3);
            newVars.TargetVarIndexes.Should().BeEquivalentTo(0, 2);
        }


        [Fact]
        public void SwapUse_when_input_set_will_be_empty_throws()
        {
            var vars = new SupervisedSetVariableIndexes(new[] { 1 }, new[] { 0 });

            vars.ChangeVariableUse(1, VariableUses.Target).Error.Should().NotBeNull();
        }
    }
}