using System;
using System.Collections.Immutable;
using System.Linq;

namespace NNLib.Common
{
    /// <summary>
    /// Used to choose input, target and ignored attributes of training data from csv file.
    /// </summary>
    public class SupervisedSetVariableIndexes
    {
        public ImmutableArray<int> Ignored { get; }
        public ImmutableArray<int> InputVarIndexes { get; }
        public ImmutableArray<int> TargetVarIndexes { get; }

        public SupervisedSetVariableIndexes(int[] inputVarIndexes, int[] targetVarIndexes, int[] ingored) : this(inputVarIndexes, targetVarIndexes)
        {
            Ignored = ingored.ToImmutableArray();
        }

        private SupervisedSetVariableIndexes(ImmutableArray<int> inputVarIndexes, ImmutableArray<int> targetVarIndexes,
            ImmutableArray<int> ignored) => (InputVarIndexes, TargetVarIndexes, Ignored) =
            (inputVarIndexes, targetVarIndexes, ignored);

        public SupervisedSetVariableIndexes(int[] inputVarIndexes, int[] targetVarIndexes)
        {
            if (inputVarIndexes.Intersect(targetVarIndexes).Any())
            {
                throw new ArgumentException("inputVarIndexes and targetVarIndexes contains the same elements");
            }

            if (inputVarIndexes.Length <= 0)
            {
                throw new ArgumentException("Input variable(s) must be set");
            }

            if (targetVarIndexes.Length <= 0)
            {
                throw new ArgumentException("Target variable(s) must be set");
            }


            Array.Sort(inputVarIndexes);
            Array.Sort(targetVarIndexes);

            if (Ignored.IsDefault)
            {
                Ignored = ImmutableArray<int>.Empty;
            }

            InputVarIndexes = inputVarIndexes.ToImmutableArray();
            TargetVarIndexes = targetVarIndexes.ToImmutableArray();
        }

        public SupervisedSetVariableIndexes ChangeVariableUse(int index, VariableUses variableUse)
        {
            if (variableUse == VariableUses.Ignore)
            {
                return IgnoreVariable(index);
            }

            ImmutableArray<int> newInputVars;
            ImmutableArray<int> newTargetVars;
            ImmutableArray<int> newIgnored = Ignored;
            if (variableUse == VariableUses.Target && InputVarIndexes.Contains(index))
            {
                newInputVars = InputVarIndexes.Remove(index);
                newTargetVars = TargetVarIndexes.Add(index);
            }
            else if (variableUse == VariableUses.Input && TargetVarIndexes.Contains(index))
            {
                newInputVars = InputVarIndexes.Add(index);
                newTargetVars = TargetVarIndexes.Remove(index);
            }
            else if (Ignored.Contains(index))
            {
                newIgnored = newIgnored.Remove(index);
                if (variableUse == VariableUses.Input)
                {
                    newInputVars = InputVarIndexes.Add(index);
                    newTargetVars = TargetVarIndexes;
                }
                else
                {
                    newInputVars = InputVarIndexes;
                    newTargetVars = TargetVarIndexes.Add(index);
                }
            }
            else if ((variableUse == VariableUses.Input && InputVarIndexes.Contains(index)) ||
                     (variableUse == VariableUses.Target && TargetVarIndexes.Contains(index)))
            {
                newInputVars = InputVarIndexes;
                newTargetVars = TargetVarIndexes;
            }
            else
            {
                throw new Exception();
            }

            return new SupervisedSetVariableIndexes(newInputVars.ToArray(), newTargetVars.ToArray(), newIgnored.ToArray());
        }

        private SupervisedSetVariableIndexes IgnoreVariable(int index)
        {
            ImmutableArray<int> newInputVars;
            ImmutableArray<int> newTargetVars;
            ImmutableArray<int> newIgnored = Ignored;
            if (InputVarIndexes.Contains(index))
            {
                if (InputVarIndexes.Length == 1)
                {
                    throw new InvalidOperationException("Input variables must be set");
                }
                newInputVars = InputVarIndexes.Remove(index);
                newTargetVars = TargetVarIndexes;
            }
            else if (TargetVarIndexes.Contains(index))
            {
                if (TargetVarIndexes.Length == 1)
                {
                    throw new InvalidOperationException("Target variables must be set");
                }
                newInputVars = InputVarIndexes;
                newTargetVars = TargetVarIndexes.Remove(index);
            }
            else if (Ignored.Contains(index))
            {
                return new SupervisedSetVariableIndexes(InputVarIndexes.ToArray(), TargetVarIndexes.ToArray(), Ignored.ToArray());
            }
            else
            {
                throw new Exception($"Ignored array doesn't contain {index} index");
            }

            newIgnored = newIgnored.Add(index);

            return new SupervisedSetVariableIndexes(newInputVars.ToArray(), newTargetVars.ToArray(), newIgnored.ToArray());
        }

        public SupervisedSetVariableIndexes Clone() => new SupervisedSetVariableIndexes(InputVarIndexes, TargetVarIndexes, Ignored);
    }
}