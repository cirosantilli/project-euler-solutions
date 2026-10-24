import «.formalsw».solvers.«70.py.ast»
import ProjectEulerSolutions.Lib.TotientSieveCalls

/-! Instantiate the shared linear-sieve source proof for the totient-permutation solver. -/
namespace solvers.«70».sieve
open FormalSW.PyAst HeapCalls
set_option Elab.async false

abbrev program := Solvers.P70Ast.module
export ProjectEulerSolutions.TotientSieveCalls (number entries)

theorem implementation :
    ProjectEulerSolutions.TotientSieveCalls.Implements program "compute_phi_linear" :=
  ⟨((findFunctionIn program.body "compute_phi_linear").getD ([], [])).2, rfl, rfl⟩

/-- Every signed bound returns the complete list of totients for its allocated interval. -/
theorem result (limit : Int) (base : Mutable.Heap) :
    ∃ finalHeap, ReturnsValue program "compute_phi_linear" [.scalar (.int limit)] base
      (.ref (base.length + 1)) finalHeap ∧
      finalHeap[base.length + 1]? = some
        (.list (entries (limit + 1).toNat (fun n => number n.totient))) :=
  ProjectEulerSolutions.TotientSieveCalls.result "compute_phi_linear" implementation limit base

end solvers.«70».sieve
