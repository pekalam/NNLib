using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace NNLib
{
    public class Lockable
    {
        protected void CheckIsLocked() { if (Locked) throw new ObjectLockedException("Layer locked"); }

        internal virtual void Lock([CallerMemberName] string caller = "")
        {
            CheckIsLocked();
            Trace.WriteLine("Layer obj LOCKED by " + caller);
            Locked = true;
        }

        internal virtual void Unlock([CallerMemberName] string caller = "")
        {
            Trace.WriteLine("Layer obj UNLOCKED by " + caller);
            Locked = false;
        }

        public bool Locked { get; private set; }
    }

    public class Lockable<T> : Lockable where T : Lockable
    {
        private List<T> _children = null;

        protected void SetLockableChildren(List<T> lockables) => _children = lockables;

        internal override void Lock(string caller = "")
        {
            base.Lock(caller);
            if (_children != null)
            {
                foreach (var child in _children)
                {
                    child.Lock();
                }
            }
        }

        internal override void Unlock(string caller = "")
        {
            base.Unlock(caller);
            if (_children != null)
            {
                foreach (var child in _children)
                {
                    child.Unlock();
                }
            }
        }
    }
}