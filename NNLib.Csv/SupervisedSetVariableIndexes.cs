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
        private readonly ImmutableArray<int> _ignored;
        private readonly ImmutableArray<int> _inputVarIndexes;
        private readonly ImmutableArray<int> _targetVarIndexes;

        public ImmutableArray<int> Ignored => Error == null ? _ignored : throw new ArgumentException(Error);

        public ImmutableArray<int> InputVarIndexes => Error == null ? _inputVarIndexes : throw new ArgumentException(Error);

        public ImmutableArray<int> TargetVarIndexes => Error == null ? _targetVarIndexes : throw new ArgumentException(Error);

        public SupervisedSetVariableIndexes(int[] inputVarIndexes, int[] targetVarIndexes, int[] ingored) : this(inputVarIndexes, targetVarIndexes)
        {
            _ignored = ingored.ToImmutableArray();
        }

        private SupervisedSetVariableIndexes(ImmutableArray<int> inputVarIndexes, ImmutableArray<int> targetVarIndexes,
            ImmutableArray<int> ignored) => (_inputVarIndexes, _targetVarIndexes, _ignored) =
            (inputVarIndexes, targetVarIndexes, ignored);

        public SupervisedSetVariableIndexes(int[] inputVarIndexes, int[] targetVarIndexes)
        {
            if (inputVarIndexes.Intersect(targetVarIndexes).Any())
            {
                Error = "inputVarIndexes and targetVarIndexes contains the same elements";
            }

            if (inputVarIndexes.Length <= 0)
            {
                Error = "Input variable(s) must be set";
            }

            if (targetVarIndexes.Length <= 0)
            {
                Error = "Target variable(s) must be set";
            }


            Array.Sort(inputVarIndexes);
            Array.Sort(targetVarIndexes);

            if (_ignored.IsDefault)
            {
                _ignored = ImmutableArray<int>.Empty;
            }

            _inputVarIndexes = inputVarIndexes.ToImmutableArray();
            _targetVarIndexes = targetVarIndexes.ToImmutableArray();
        }

        public string? Error { get; private set; }

        public SupervisedSetVariableIndexes ChangeVariableUse(int index, VariableUses variableUse)
        {
            if (variableUse == VariableUses.Ignore)
            {
                return IgnoreVariable(index);
            }

            ImmutableArray<int> newInputVars;
            ImmutableArray<int> newTargetVars;
            ImmutableArray<int> new_ignored = _ignored;
            if (variableUse == VariableUses.Target && _inputVarIndexes.Contains(index))
            {
                newInputVars = _inputVarIndexes.Remove(index);
                newTargetVars = _targetVarIndexes.Add(index);
            }
            else if (variableUse == VariableUses.Input && _targetVarIndexes.Contains(index))
            {
                newInputVars = _inputVarIndexes.Add(index);
                newTargetVars = _targetVarIndexes.Remove(index);
            }
            else if (_ignored.Contains(index))
            {
                new_ignored = new_ignored.Remove(index);
                if (variableUse == VariableUses.Input)
                {
                    newInputVars = _inputVarIndexes.Add(index);
                    newTargetVars = _targetVarIndexes;
                }
                else
                {
                    newInputVars = _inputVarIndexes;
                    newTargetVars = _targetVarIndexes.Add(index);
                }
            }
            else if ((variableUse == VariableUses.Input && _inputVarIndexes.Contains(index)) ||
                     (variableUse == VariableUses.Target && _targetVarIndexes.Contains(index)))
            {
                newInputVars = _inputVarIndexes;
                newTargetVars = _targetVarIndexes;
            }
            else
            {
                throw new Exception();
            }

            return new SupervisedSetVariableIndexes(newInputVars.ToArray(), newTargetVars.ToArray(), new_ignored.ToArray());
        }

        private SupervisedSetVariableIndexes IgnoreVariable(int index)
        {
            ImmutableArray<int> newInputVars;
            ImmutableArray<int> newTargetVars;
            ImmutableArray<int> new_ignored = _ignored;
            if (_inputVarIndexes.Contains(index))
            {
                if (_inputVarIndexes.Length == 1)
                {
                    Error = "Input variables must be set";
                }
                newInputVars = _inputVarIndexes.Remove(index);
                newTargetVars = _targetVarIndexes;
            }
            else if (_targetVarIndexes.Contains(index))
            {
                if (_targetVarIndexes.Length == 1)
                {
                    Error = "Target variables must be set";
                }
                newInputVars = _inputVarIndexes;
                newTargetVars = _targetVarIndexes.Remove(index);
            }
            else if (_ignored.Contains(index))
            {
                return new SupervisedSetVariableIndexes(_inputVarIndexes.ToArray(), _targetVarIndexes.ToArray(), _ignored.ToArray());
            }
            else
            {
                throw new Exception($"_ignored array doesn't contain {index} index");
            }

            new_ignored = new_ignored.Add(index);

            return new SupervisedSetVariableIndexes(newInputVars.ToArray(), newTargetVars.ToArray(), new_ignored.ToArray());
        }

        public SupervisedSetVariableIndexes Clone() => new SupervisedSetVariableIndexes(_inputVarIndexes, _targetVarIndexes, _ignored);
    }
}