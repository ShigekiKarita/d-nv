
import std.range;
import std.meta;
import std.array;
import std.traits;

import std.stdio;


struct Param(string args) {
  mixin("private void _fun_(" ~ args ~ ") {}");
  alias Args = AliasSeq!(Parameters!(_fun_));
}


template isArgTypesEqualT(string kargs, Args ...) {
  enum bool value = is(Param!kargs.Args == AliasSeq!Args);
}

bool isEqualArgTypes(string kargs, Args ...)(Args args) {
  return isArgTypesEqualT!(kargs, Args).value;
}

void assertEqualArgTypes(string kargs, Args ...)(Args args) {
  static assert(isArgTypesEqualT!(kargs, Args).value,
                "(%s) is not equal to %s".format(kargs, Args.stringof));
}


unittest {
  int i = 0;
  float[] f = [1f, 2f];
  const char[] c = ['a'];
  assert(isEqualArgTypes!q{int i, float* f, const(char)* c}(i, f.ptr, c.ptr));
  assert(!isEqualArgTypes!q{float i, float* f, const(char)* c}(i, f.ptr, c.ptr));
  assertEqualArgTypes!q{int i, float* f, const(char)* c}(i, f.ptr, c.ptr);
  static assert(!__traits(compiles, assertArgTypesEqual!q{float i, float* f, const(char)* c}(i, f.ptr, c.ptr)));
}


template AssignableArgTypesT(string kargs, Args ...) {
  alias KArgs = Param!kargs.Args;
  enum bool value = function() {
    bool ok = true;
    foreach (i, a; Args) {
      ok &= isAssignable!(KArgs[i], Args[i]);
    }
    return ok;
  }();
}

bool isAssignableArgTypes(string kargs, Args ...)(Args args) {
  return AssignableArgTypesT!(kargs, Args).value;
}

void assertAssignableArgTypes(string kargs, Args ...)(Args args) {
  static assert(AssignableArgTypesT!(kargs, Args).value,
                "(%s) is not AssignableArgTypes to %s".format(kargs, Args.stringof));
}

unittest {
  int i = 0;
  float[] f = [1f, 2f];
  const char[] c = ['a'];
  assert(isAssignableArgTypes!q{int i, float* f, const(char)* c}(i, f.ptr, c.ptr));
  assert(!isAssignableArgTypes!q{int i, float* f, char* c}(i, f.ptr, c.ptr));
  assertAssignableArgTypes!q{int i, const(float)* f, const(char)* c}(i, f.ptr, c.ptr);
  static assert(!__traits(compiles, assertAssignableArgs!q{int i, const(float)* f, char* c}(i, f.ptr, c.ptr)));
}
