module dnv.typechecker;

import std.string : format, split;
import std.meta;
import std.traits;


struct Param(string args) {
  mixin("private void _fun_(" ~ args ~ ") {}");
  alias Args = AliasSeq!(Parameters!(_fun_));
}


bool predicate(alias trait, string kargs, Args ...)(Args args) {
  return trait!(kargs, Args);
}

void staticAssert(alias trait, string kargs, Args ...)(Args args) {
  static assert(trait!(kargs, Args),
                "\nleft  args:(%s)\nright args:%s\nthis right args violate the <%s> trait"
                .format(kargs, Args.stringof, trait.stringof.split("(")[0]));
}


enum bool EqualArgTypes(string kargs, Args ...) = is(Param!kargs.Args == AliasSeq!Args);

unittest {
  immutable OK = q{int i, float* f, const(char)* c};
  immutable NG = q{float i, float* f, const(char)* c};

  int i = 0;
  float[] f = [1f, 2f];
  const char[] c = ['a'];

  assert(predicate!(EqualArgTypes, OK)(i, f.ptr, c.ptr));
  assert(!predicate!(EqualArgTypes, NG)(i, f.ptr, c.ptr));

  staticAssert!(EqualArgTypes, OK)(i, f.ptr, c.ptr);
  // staticAssert!(EqualArgTypes, NG)(i, f.ptr, c.ptr);
  static assert(!__traits(compiles, staticAssert!(EqualArgTypes, NG)(i, f.ptr, c.ptr)));
}

enum bool isCudaAssignable(L, R) = function() {
  static if (is(typeof({R.Storage s = null;}))) {
    return isAssignable!(L, R.Storage);
  } else {
    return isAssignable!(L, R);
  }
}();

enum bool AssignableArgTypes(string kargs, Args ...) = function() {
  alias KArgs = Param!kargs.Args;
  if (KArgs.length != Args.length) {
    return false;
  }
  
  bool ok = true;
  foreach (i, _; Args) {
    ok &= isCudaAssignable!(KArgs[i], Args[i]);
    if (!ok) {
      return false;
    }
  }
  return true;
}();

unittest {
  int i = 0;
  float[] f = [1f, 2f];
  const char[] c = ['a'];
  immutable OK = q{int i, float* f, const(char)* c};
  immutable NG = q{int i, float* f, char* c};

  assert(predicate!(AssignableArgTypes, OK)(i, f.ptr, c.ptr));
  assert(!predicate!(AssignableArgTypes, NG)(i, f.ptr, c.ptr));

  staticAssert!(AssignableArgTypes, OK)(i, f.ptr, c.ptr);
  // staticAssert!(AssignableArgTypes, NG)(i, f.ptr, c.ptr);
  static assert(!__traits(compiles, staticAssert!(AssignableArgTypes, NG)(i, f.ptr, c.ptr)));
}

unittest {
  import dnv.storage;
  import dnv.compiler;
  static assert(isCudaAssignable!(float*, Array!float));
}
