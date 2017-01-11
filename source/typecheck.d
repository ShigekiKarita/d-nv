import std.string : format, split;
import std.meta;
import std.traits;


struct Param(string args) {
  mixin("private void _fun_(" ~ args ~ ") {}");
  alias Args = AliasSeq!(Parameters!(_fun_));
}


bool predicate(alias policy, string kargs, Args ...)(Args args) {
  return policy!(kargs, Args);
}

void staticAssert(alias policy, string kargs, Args ...)(Args args) {
  static assert(policy!(kargs, Args),
                "\nleft  args:(%s)\nright args:%s\nthis right args violate the <%s> policy"
                .format(kargs, Args.stringof, policy.stringof.split("(")[0]));
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


enum bool AssignableArgTypes(string kargs, Args ...) = function() {
  alias KArgs = Param!kargs.Args;
  bool ok = true;
  foreach (i, a; Args) {
    ok &= isAssignable!(KArgs[i], Args[i]);
  }
  return ok;
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
