import solvers.«70.sieve.source»
import solvers.«70.signature.source»

/-! Public source termination proofs for the linear sieve and decimal histogram helpers. -/
namespace solvers.«70»
open FormalSW.PyAst
set_option Elab.async false

theorem compute_phi_linear.«Returns» (limit : Int) :
    (HeapCalls.call program "compute_phi_linear" [.scalar (.int limit)] []).Returns := by
  obtain ⟨finalHeap, returned, _⟩ := sieve.result limit []
  exact returned.returns

theorem compute_phi_linear.«Supported» (limit : Int) :
    (HeapCalls.call program "compute_phi_linear" [.scalar (.int limit)] []).Supported :=
  (compute_phi_linear.«Returns» limit).supported

end solvers.«70»
