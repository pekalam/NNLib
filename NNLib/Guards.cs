using System;
using System.Runtime.CompilerServices;

[assembly: InternalsVisibleTo("NNLib")]
[assembly: InternalsVisibleTo("NNLib.Csv")]
namespace NNLib
{
    internal class Guards
    {
        private static readonly Guards Guard = new Guards();

        private Guards()
        {
        }

        public Guards GtZero(int num, string msg = "")
        {
            if (num <= 0)
            {
                throw new ArgumentException(msg);
            }

            return this;
        }

        public Guards NotNull(object obj, string msg = "", string memberName = "")
        {
            if (obj == null)
            {
                throw new NullReferenceException($"Null reference at {memberName} {msg}");
            }

            return this;
        }

        public static Guards _NotNull(object obj, string msg = "", [CallerMemberName] string memberName = "")
        {
            return Guard.NotNull(obj, msg, memberName);
        }

        public static Guards _GtZero(int num, string msg = "")
        {
            return Guard.GtZero(num, msg);
        }
    }
}