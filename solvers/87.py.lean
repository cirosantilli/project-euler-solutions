import solvers.«87.py.lean.d».Preparation

namespace solvers.«87».sieve_primes
open FormalSW.PyAst
set_option Elab.async false

theorem «Returns» (n : Int) :
    (HeapCalls.call program "sieve_primes" [.scalar (.int n)] []).Returns := by
  obtain ⟨_, _, returned, _⟩ := result n []
  exact returned.returns

theorem «Supported» (n : Int) :
    (HeapCalls.call program "sieve_primes" [.scalar (.int n)] []).Supported :=
  («Returns» n).supported

end solvers.«87».sieve_primes
