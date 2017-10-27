module dnv.compiler;

import std.conv;
import std.string;

import dnv.driver;
import dnv.typechecker;
import dnv.error;


struct Code {
    /*
      FIXME: template support
      - see: /usr/local/cuda/samples/0_Simple/simpleTemplates_nvrtc
    */
    immutable qualifier = `extern "C" __global__ `;
    immutable returnType = "void ";
    const string name;
    const string args;
    const string source;

    this(in string nameStr, in string argumentsStr, in string bodyStr) {
        name = nameStr;
        args = argumentsStr;
        source = qualifier ~ returnType ~ nameStr ~
            "(" ~ args ~ ")" ~
            "{" ~ bodyStr ~ "}";
    }
}

unittest {
    auto saxpy = Code(
    "saxpy", "float *A, float *B, float *C, int numElements", `
      int i = blockDim.x * blockIdx.x + threadIdx.x;
      if (i < numElements) C[i] = A[i] + B[i];
    `);
    assert(saxpy.source ==
   `extern "C" __global__ void saxpy(float *A, float *B, float *C, int numElements){
      int i = blockDim.x * blockIdx.x + threadIdx.x;
      if (i < numElements) C[i] = A[i] + B[i];
    }`);
}


struct UnsafeCompiler {
    static void build(void* vfunc, Code c) {
        check(compile(vfunc, c.name, c.source));
    }
    void assertArgs(Args...)(Args args) {}
}

struct StaticCompiler(Code c) {
    immutable Code code = c;
    immutable cargs = c.args;
    static void build(void* vfunc, Code c) {
        check(compile(vfunc, c.name, c.source));
    }
    void assertArgs(Args...)(Args targs) {
        // FIXME: cannot call
        staticAssert!(AssignableArgTypes, cargs)(targs);
    }
}

struct SimpleLauncher {
    uint[3] grids = [256, 1, 1];
    uint[3] blocks;

    void setup(Args...)(Args targs) {
        uint bx = to!uint((grids[0] + targs[0].length - 1) / grids[0]);
        blocks = [bx, 1, 1];
    }
}
